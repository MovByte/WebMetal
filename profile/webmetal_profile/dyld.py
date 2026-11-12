"""Read-only access to an arm64 dyld shared cache that is split into subcaches.

Structure layouts follow dyld's `dyld_cache_format.h` (`dyld_cache_header`,
`dyld_cache_mapping_and_slide_info`, `dyld_subcache_entry`, `dyld_cache_slide_info2`,
`dyld_cache_local_symbols_info`, `dyld_cache_local_symbols_entry_64`,
`dyld_cache_image_text_info`) and the Mach-O headers `<mach-o/loader.h>` and
`<mach-o/nlist.h>`. Every structure is checked against the file it is read from; a value
outside the documented form raises `CacheError` instead of being guessed around.
"""

import bisect
import mmap
import struct
from pathlib import Path


class CacheError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CacheError(message)


def _layout(fields):
    """Offsets of a C structure with natural alignment (all members here are scalars)."""
    sizes = {'I': 4, 'Q': 8}
    offsets, at = {}, 0
    for name, kind in fields:
        size = int(kind[:-1]) if kind.endswith('s') else sizes[kind]
        align = 1 if kind.endswith('s') else size
        at = (at + align - 1) // align * align
        offsets[name] = (at, kind)
        at += size
    return offsets, at


# struct dyld_cache_header, in declaration order. `formatVersion` and the following
# bitfields share one uint32_t.
HEADER, HEADER_SIZE = _layout([
    ('magic', '16s'), ('mappingOffset', 'I'), ('mappingCount', 'I'), ('imagesOffsetOld', 'I'),
    ('imagesCountOld', 'I'), ('dyldBaseAddress', 'Q'), ('codeSignatureOffset', 'Q'),
    ('codeSignatureSize', 'Q'), ('slideInfoOffsetUnused', 'Q'), ('slideInfoSizeUnused', 'Q'),
    ('localSymbolsOffset', 'Q'), ('localSymbolsSize', 'Q'), ('uuid', '16s'), ('cacheType', 'Q'),
    ('branchPoolsOffset', 'I'), ('branchPoolsCount', 'I'), ('dyldInCacheMH', 'Q'),
    ('dyldInCacheEntry', 'Q'), ('imagesTextOffset', 'Q'), ('imagesTextCount', 'Q'),
    ('patchInfoAddr', 'Q'), ('patchInfoSize', 'Q'), ('otherImageGroupAddrUnused', 'Q'),
    ('otherImageGroupSizeUnused', 'Q'), ('progClosuresAddr', 'Q'), ('progClosuresSize', 'Q'),
    ('progClosuresTrieAddr', 'Q'), ('progClosuresTrieSize', 'Q'), ('platform', 'I'),
    ('formatVersionBits', 'I'), ('sharedRegionStart', 'Q'), ('sharedRegionSize', 'Q'),
    ('maxSlide', 'Q'), ('dylibsImageArrayAddr', 'Q'), ('dylibsImageArraySize', 'Q'),
    ('dylibsTrieAddr', 'Q'), ('dylibsTrieSize', 'Q'), ('otherImageArrayAddr', 'Q'),
    ('otherImageArraySize', 'Q'), ('otherTrieAddr', 'Q'), ('otherTrieSize', 'Q'),
    ('mappingWithSlideOffset', 'I'), ('mappingWithSlideCount', 'I'),
    ('dylibsPBLStateArrayAddrUnused', 'Q'), ('dylibsPBLSetAddr', 'Q'),
    ('programsPBLSetPoolAddr', 'Q'), ('programsPBLSetPoolSize', 'Q'), ('programTrieAddr', 'Q'),
    ('programTrieSize', 'I'), ('osVersion', 'I'), ('altPlatform', 'I'), ('altOsVersion', 'I'),
    ('swiftOptsOffset', 'Q'), ('swiftOptsSize', 'Q'), ('subCacheArrayOffset', 'I'),
    ('subCacheArrayCount', 'I'), ('symbolFileUUID', '16s'), ('rosettaReadOnlyAddr', 'Q'),
    ('rosettaReadOnlySize', 'Q'), ('rosettaReadWriteAddr', 'Q'), ('rosettaReadWriteSize', 'Q'),
    ('imagesOffset', 'I'), ('imagesCount', 'I'), ('cacheSubType', 'I'), ('objcOptsOffset', 'Q'),
    ('objcOptsSize', 'Q'), ('cacheAtlasOffset', 'Q'), ('cacheAtlasSize', 'Q'),
    ('dynamicDataOffset', 'Q'), ('dynamicDataMaxSize', 'Q'),
])
# struct dyld_cache_mapping_and_slide_info
MAPPING_SLIDE = struct.Struct('<6Q2I')
# struct dyld_subcache_entry (the form with fileSuffix)
SUBCACHE_ENTRY = struct.Struct('<16sQ32s')
# struct dyld_cache_image_text_info
IMAGE_TEXT = struct.Struct('<16sQII')
# struct dyld_cache_slide_info2 (fixed part)
SLIDE2 = struct.Struct('<6I2Q')
SLIDE_PAGE_ATTR_EXTRA = 0x8000
SLIDE_PAGE_ATTR_NO_REBASE = 0x4000
SLIDE_PAGE_ATTR_END = 0x8000
# struct dyld_cache_local_symbols_info / dyld_cache_local_symbols_entry_64
LOCAL_SYMBOLS_INFO = struct.Struct('<6I')
LOCAL_SYMBOLS_ENTRY64 = struct.Struct('<QII')
# <mach-o/loader.h>
MH_MAGIC_64 = 0xfeedfacf
LC_SEGMENT_64 = 0x19
LC_SYMTAB = 0x2
LC_UUID = 0x1b
SECTION_64 = struct.Struct('<16s16sQQ8I')
SEGMENT_64 = struct.Struct('<16sQQQQiiII')
# <mach-o/nlist.h>
NLIST_64 = struct.Struct('<IBBHQ')
N_STAB, N_TYPE, N_SECT = 0xe0, 0x0e, 0x0e


def header_field(data, name):
    offset, kind = HEADER[name]
    return struct.unpack_from('<' + kind, data, offset)[0]


class SlideInfoV2:
    def __init__(self, data, offset, size, mapping_address, mapping_size):
        require(size >= SLIDE2.size, 'slide info shorter than dyld_cache_slide_info2')
        (version, page_size, starts_offset, starts_count, extras_offset, extras_count,
         delta_mask, value_add) = SLIDE2.unpack_from(data, offset)
        require(version == 2, f'slide info version {version}; only version 2 is implemented')
        require(page_size in (4096, 16384), f'slide page size {page_size}')
        require(starts_count * page_size >= mapping_size, 'slide page starts do not cover mapping')
        require(delta_mask and (delta_mask & -delta_mask) >= 4, 'slide delta mask')
        self.page_size = page_size
        self.delta_mask = delta_mask
        self.value_mask = ~delta_mask & (1 << 64) - 1
        self.delta_shift = (delta_mask & -delta_mask).bit_length() - 1 - 2
        self.value_add = value_add
        self.mapping_address = mapping_address
        self.starts = struct.unpack_from(f'<{starts_count}H', data, offset + starts_offset)
        self.extras = struct.unpack_from(f'<{extras_count}H', data, offset + extras_offset)

    def decode(self, raw):
        value = raw & self.value_mask
        return value + self.value_add if value else 0


class Mapping:
    __slots__ = ('address', 'size', 'end', 'data', 'file_offset', 'slide', 'max_prot', 'init_prot',
                 'file_name', '_chains')

    def __init__(self, address, size, data, file_offset, slide, max_prot, init_prot, file_name):
        self.address, self.size, self.end = address, size, address + size
        self.data, self.file_offset, self.slide = data, file_offset, slide
        self.max_prot, self.init_prot, self.file_name = max_prot, init_prot, file_name
        self._chains = {}

    def raw(self, address, size):
        at = self.file_offset + address - self.address
        return self.data[at:at + size]

    def rebases(self, page_index):
        """Offsets (from the slide page start) of every rebased pointer in one slide page."""
        if page_index in self._chains:
            return self._chains[page_index]
        info, found = self.slide, {}
        start = info.starts[page_index]
        heads = []
        if start == SLIDE_PAGE_ATTR_NO_REBASE:
            pass
        elif start & SLIDE_PAGE_ATTR_EXTRA:
            index = start & 0x3fff
            while True:
                extra = info.extras[index]
                heads.append((extra & 0x3fff) * 4)
                if extra & SLIDE_PAGE_ATTR_END:
                    break
                index += 1
        else:
            heads.append(start * 4)
        page_address = self.address + page_index * info.page_size
        for offset in heads:
            while True:
                require(offset + 8 <= info.page_size, 'slide chain leaves its page')
                raw, = struct.unpack('<Q', self.raw(page_address + offset, 8))
                found[offset] = info.decode(raw)
                delta = (raw & info.delta_mask) >> info.delta_shift
                if not delta:
                    break
                offset += delta
        self._chains[page_index] = found
        return found


class Image:
    def __init__(self, cache, path, address, text_size, uuid):
        self.cache, self.path, self.address, self.text_size, self.uuid = cache, path, address, text_size, uuid
        self.name = path.rsplit('/', 1)[-1]
        self._parsed = False
        self._symbols = None

    def _parse(self):
        if self._parsed:
            return
        cache = self.cache
        magic, _, _, _, ncmds, sizeofcmds, _, _ = struct.unpack('<8I', cache.read(self.address, 32))
        require(magic == MH_MAGIC_64, f'{self.path}: not a 64-bit Mach-O header')
        commands = cache.read(self.address + 32, sizeofcmds)
        self.segments, self.sections, self.symtab, at = {}, {}, None, 0
        for _ in range(ncmds):
            cmd, size = struct.unpack_from('<2I', commands, at)
            require(size >= 8 and at + size <= sizeofcmds, f'{self.path}: load command bounds')
            if cmd == LC_SEGMENT_64:
                name, vmaddr, vmsize, fileoff, filesize, _, _, nsects, _ = SEGMENT_64.unpack_from(commands, at + 8)
                segment = name.rstrip(b'\0').decode()
                self.segments[segment] = (vmaddr, vmsize, fileoff, filesize)
                for i in range(nsects):
                    fields = SECTION_64.unpack_from(commands, at + 8 + SEGMENT_64.size + i * SECTION_64.size)
                    section = fields[0].rstrip(b'\0').decode()
                    self.sections[(fields[1].rstrip(b'\0').decode(), section)] = (fields[2], fields[3])
            elif cmd == LC_SYMTAB:
                self.symtab = struct.unpack_from('<4I', commands, at + 8)
            elif cmd == LC_UUID:
                require(commands[at + 8:at + 24] == self.uuid, f'{self.path}: LC_UUID differs from cache record')
            at += size
        require('__TEXT' in self.segments and self.segments['__TEXT'][0] == self.address,
                f'{self.path}: __TEXT segment does not start at the image address')
        self._parsed = True

    def section(self, segment_prefix, name):
        """All sections with this name in segments whose name starts with segment_prefix."""
        self._parse()
        return [(seg, addr, size) for (seg, sect), (addr, size) in sorted(self.sections.items())
                if sect == name and seg.startswith(segment_prefix)]

    def segment(self, name):
        self._parse()
        return self.segments[name]

    def contains_text(self, address):
        return self.address <= address < self.address + self.text_size

    @property
    def symbols(self):
        """name -> address for every defined (local or global) section symbol of this image."""
        if self._symbols is None:
            self._parse()
            table = {}
            for name, address in self.cache.local_symbols(self):
                table.setdefault(name, address)
            if self.symtab:
                symoff, nsyms, stroff, strsize = self.symtab
                link_vmaddr, _, link_fileoff, _ = self.segments['__LINKEDIT']
                entries = self.cache.read(link_vmaddr + symoff - link_fileoff, nsyms * NLIST_64.size)
                strings = self.cache.read(link_vmaddr + stroff - link_fileoff, strsize)
                for i in range(nsyms):
                    strx, ntype, _, _, value = NLIST_64.unpack_from(entries, i * NLIST_64.size)
                    if ntype & N_STAB or ntype & N_TYPE != N_SECT:
                        continue
                    end = strings.index(b'\0', strx)
                    table.setdefault(strings[strx:end].decode('utf-8', 'replace'), value)
            self._symbols = table
            ordered = sorted((address, name) for name, address in table.items())
            self._by_address = ([a for a, _ in ordered], [n for _, n in ordered])
        return self._symbols

    def symbolize(self, address):
        """(name, offset) of the closest symbol at or below address."""
        self.symbols
        addresses, names = self._by_address
        i = bisect.bisect_right(addresses, address) - 1
        if i < 0:
            return None, None
        return names[i], address - addresses[i]


class Cache:
    """A dyld shared cache: the main file, its subcaches and its .symbols file."""

    def __init__(self, main_path):
        main_path = Path(main_path)
        self.path = main_path
        self._files = []
        main = self._open(main_path)
        require(main[:7] == b'dyld_v1' and main[:16].rstrip(b'\0').endswith(b'arm64'),
                'not an arm64 dyld shared cache')
        mapping_offset = header_field(main, 'mappingOffset')
        require(mapping_offset >= HEADER['subCacheArrayCount'][0] + 4,
                'cache header predates subcache arrays')
        self.header_size = mapping_offset
        self.uuid = header_field(main, 'uuid')
        self.region = header_field(main, 'sharedRegionStart')
        self.region_size = header_field(main, 'sharedRegionSize')
        self.platform = header_field(main, 'platform')
        self.os_version = header_field(main, 'osVersion')
        self.main = main
        self.mappings = []
        self._add_mappings(main, main_path.name)
        count = header_field(main, 'subCacheArrayCount')
        at = header_field(main, 'subCacheArrayOffset')
        require(mapping_offset > HEADER['cacheSubType'][0], 'cache uses v1 subcache entries')
        self.subcaches = []
        for i in range(count):
            uuid, vm_offset, suffix = SUBCACHE_ENTRY.unpack_from(main, at + i * SUBCACHE_ENTRY.size)
            suffix = suffix.rstrip(b'\0').decode()
            path = main_path.with_name(main_path.name + suffix)
            data = self._open(path)
            require(header_field(data, 'uuid') == uuid, f'{path.name}: UUID differs from subcache entry')
            self._add_mappings(data, path.name)
            self.subcaches.append((suffix, uuid, vm_offset))
        self.mappings.sort(key=lambda m: m.address)
        self._starts = [m.address for m in self.mappings]
        for a, b in zip(self.mappings, self.mappings[1:]):
            require(a.end <= b.address, 'overlapping cache mappings')
        symbol_uuid = header_field(main, 'symbolFileUUID')
        self.symbols_path = main_path.with_name(main_path.name + '.symbols')
        self._symbols_file = self._open(self.symbols_path)
        require(header_field(self._symbols_file, 'uuid') == symbol_uuid, '.symbols file UUID mismatch')
        self._local_index = None
        self.images = []
        at, count = header_field(main, 'imagesTextOffset'), header_field(main, 'imagesTextCount')
        for i in range(count):
            uuid, address, size, path_offset = IMAGE_TEXT.unpack_from(main, at + i * IMAGE_TEXT.size)
            path = main[path_offset:main.find(b'\0', path_offset)].decode()
            self.images.append(Image(self, path, address, size, uuid))
        self.images.sort(key=lambda image: image.address)
        self._image_starts = [image.address for image in self.images]

    def _open(self, path):
        stream = open(path, 'rb')
        data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self._files.append((stream, data))
        return data

    def _add_mappings(self, data, name):
        at, count = header_field(data, 'mappingWithSlideOffset'), header_field(data, 'mappingWithSlideCount')
        require(count > 0, f'{name}: no mapping-with-slide records')
        for i in range(count):
            address, size, file_offset, slide_offset, slide_size, _, max_prot, init_prot = \
                MAPPING_SLIDE.unpack_from(data, at + i * MAPPING_SLIDE.size)
            if not size:
                continue
            require(file_offset + size <= len(data), f'{name}: mapping exceeds file')
            slide = SlideInfoV2(data, slide_offset, slide_size, address, size) if slide_size else None
            self.mappings.append(Mapping(address, size, data, file_offset, slide, max_prot, init_prot, name))

    # -- address space ----------------------------------------------------------------------
    def mapping(self, address):
        i = bisect.bisect_right(self._starts, address) - 1
        if i >= 0 and self.mappings[i].address <= address < self.mappings[i].end:
            return self.mappings[i]
        return None

    def read(self, address, size):
        m = self.mapping(address)
        require(m is not None and address + size <= m.end, f'unbacked cache range {address:#x}+{size:#x}')
        return m.raw(address, size)

    def u32(self, address):
        return struct.unpack('<I', self.read(address, 4))[0]

    def i32(self, address):
        return struct.unpack('<i', self.read(address, 4))[0]

    def u64(self, address):
        return struct.unpack('<Q', self.read(address, 8))[0]

    def pointer(self, address):
        """A pointer-sized field. In a slid mapping the field must be a rebase location or zero."""
        m = self.mapping(address)
        require(m is not None and address + 8 <= m.end, f'unbacked pointer at {address:#x}')
        raw, = struct.unpack('<Q', m.raw(address, 8))
        if m.slide is None:
            require(raw == 0, f'pointer field at {address:#x} lies in an unslid mapping')
            return 0
        page, offset = divmod(address - m.address, m.slide.page_size)
        rebases = m.rebases(page)
        if offset in rebases:
            return rebases[offset]
        require(raw == 0, f'pointer field at {address:#x} is not a rebase location')
        return 0

    def materialize(self, address, size):
        """Bytes of [address, address+size) as the loaded (unslid) process sees them."""
        m = self.mapping(address)
        require(m is not None and address + size <= m.end, f'unbacked cache range {address:#x}')
        data = bytearray(m.raw(address, size))
        if m.slide is not None:
            ps = m.slide.page_size
            first = (address - m.address) // ps
            last = (address + size - 1 - m.address) // ps
            for page in range(first, last + 1):
                base = m.address + page * ps
                for offset, value in m.rebases(page).items():
                    at = base + offset - address
                    if 0 <= at and at + 8 <= size:
                        struct.pack_into('<Q', data, at, value)
                    else:
                        require(at + 8 <= 0 or at >= size, 'rebased pointer straddles a materialized range')
        return bytes(data)

    def cstring(self, address, limit=4096):
        m = self.mapping(address)
        require(m is not None, f'unbacked string at {address:#x}')
        data = m.raw(address, min(limit, m.end - address))
        end = data.find(b'\0')
        require(end >= 0, f'unterminated string at {address:#x}')
        return data[:end].decode('utf-8')

    # -- images and symbols -----------------------------------------------------------------
    def image(self, suffix):
        found = [image for image in self.images if image.path.endswith('/' + suffix) or image.path == suffix]
        require(len(found) == 1, f'expected exactly one image named {suffix}, found {len(found)}')
        return found[0]

    def image_at(self, address):
        """The image whose __TEXT contains address, else None."""
        i = bisect.bisect_right(self._image_starts, address) - 1
        if i >= 0 and self.images[i].contains_text(address):
            return self.images[i]
        return None

    def local_symbols(self, image):
        if self._local_index is None:
            data = self._symbols_file
            base = header_field(data, 'localSymbolsOffset')
            require(base, '.symbols file has no local symbols')
            nlist_offset, nlist_count, strings_offset, strings_size, entries_offset, entries_count = \
                LOCAL_SYMBOLS_INFO.unpack_from(data, base)
            index = {}
            for i in range(entries_count):
                dylib, start, count = LOCAL_SYMBOLS_ENTRY64.unpack_from(data, base + entries_offset + i * LOCAL_SYMBOLS_ENTRY64.size)
                index[dylib] = (start, count)
            self._local_index = (base, nlist_offset, nlist_count, strings_offset, strings_size, index)
        base, nlist_offset, nlist_count, strings_offset, strings_size, index = self._local_index
        entry = index.get(image.address - self.region)
        if entry is None:
            return []
        start, count = entry
        require(start + count <= nlist_count, 'local symbol range exceeds nlist table')
        data, out = self._symbols_file, []
        strings = base + strings_offset
        for i in range(start, start + count):
            strx, ntype, _, _, value = NLIST_64.unpack_from(data, base + nlist_offset + i * NLIST_64.size)
            if ntype & N_STAB or ntype & N_TYPE != N_SECT:
                continue
            require(strx < strings_size, 'local symbol name out of range')
            end = data.find(b'\0', strings + strx)
            out.append((data[strings + strx:end].decode('utf-8', 'replace'), value))
        return out

    def symbol(self, image_suffix, name):
        table = self.image(image_suffix).symbols
        require(name in table, f'{image_suffix}: no symbol {name}')
        return table[name]

    def symbolize(self, address):
        image = self.image_at(address)
        if image is None:
            return None, None, None
        name, offset = image.symbolize(address)
        return image, name, offset
