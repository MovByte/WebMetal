"""Objective-C metadata of images in a dyld shared cache (arm64, no pointer authentication).

Layouts follow objc4's `objc-runtime-new.h` (`objc_class`, `class_ro_t`, `method_list_t` in
its pointer and relative forms, `relative_list_list_t`, `ivar_list_t`, `protocol_t`,
`protocol_list_t`, `category_t`) and the shared cache's ObjC optimization header
(`relativeMethodSelectorBaseAddressOffset`, `headerInfoROCacheOffset`). Every list is bounded
and every string is checked, so a misread structure fails instead of producing names.
"""

import struct

from .dyld import CacheError, header_field, require

FAST_DATA_MASK = 0x00007ffffffffff8          # class_data_bits_t, __LP64__ without ptrauth
RO_META = 1 << 0
SMALL_METHOD_LIST = 0x80000000
RELATIVE_SELECTORS_DIRECT = 0x40000000
METHOD_LIST_FLAGS = 0xffff0003
LIST_BOUND = 1 << 14


class Method:
    __slots__ = ('selector', 'types', 'imp', 'image_index')

    def __init__(self, selector, types, imp, image_index):
        self.selector, self.types, self.imp, self.image_index = selector, types, imp, image_index


class Ivar:
    __slots__ = ('name', 'type', 'offset', 'size', 'alignment')

    def __init__(self, name, type_, offset, size, alignment):
        self.name, self.type, self.offset, self.size, self.alignment = name, type_, offset, size, alignment


class Class:
    def __init__(self, runtime, address):
        cache = runtime.cache
        self.address = address
        self.isa = cache.pointer(address)
        self.superclass = cache.pointer(address + 8)
        bits = cache.pointer(address + 32)
        ro = bits & FAST_DATA_MASK
        require(ro, f'class {address:#x} has no class_ro_t')
        self.ro = ro
        self.flags, self.instance_start, self.instance_size = struct.unpack('<3I', cache.read(ro, 12))
        self.name = cache.cstring(cache.pointer(ro + 24))
        require(self.name and self.instance_size <= 1 << 20 and self.instance_start <= self.instance_size,
                f'implausible class_ro_t at {ro:#x}')
        self.is_meta = bool(self.flags & RO_META)
        self.method_lists = runtime.method_lists(cache.pointer(ro + 32))
        self.protocols = runtime.protocol_list(cache.pointer(ro + 40))
        self.ivars = runtime.ivar_list(cache.pointer(ro + 48))

    def own_methods(self):
        """Every (list image index, Method) in lookup order (category lists first)."""
        return [m for methods in self.method_lists for m in methods]


class Protocol:
    def __init__(self, runtime, address):
        cache = runtime.cache
        self.address = address
        self.name = cache.cstring(cache.pointer(address + 8))
        self.protocols = runtime.protocol_list(cache.pointer(address + 16))
        size, = struct.unpack('<I', cache.read(address + 64, 4))
        require(size >= 72, f'protocol {self.name}: size {size}')
        extended = cache.pointer(address + 72) if size >= 80 else 0
        lists = []
        for kind, offset in (('required', 24), ('class_required', 32), ('optional', 40), ('class_optional', 48)):
            lists.append((kind, runtime.protocol_methods(cache.pointer(address + offset))))
        # _extendedMethodTypes is one array that follows the four lists in this order.
        self.methods, index = [], 0
        for kind, methods in lists:
            for selector, types in methods:
                full = cache.cstring(cache.pointer(extended + 8 * index)) if extended else types
                self.methods.append((kind, selector, types, full))
                index += 1


class Runtime:
    def __init__(self, cache):
        self.cache = cache
        opt = cache.region + header_field(cache.main, 'objcOptsOffset')
        version, = struct.unpack('<I', cache.read(opt, 4))
        require(version == 1, f'ObjC optimization header version {version}')
        header_info_ro, = struct.unpack('<Q', cache.read(opt + 8, 8))
        self.selector_base = cache.region + struct.unpack('<Q', cache.read(opt + 48, 8))[0]
        # objc_headeropt_ro_t { uint32_t count; uint32_t entsize; header_info headers[]; }; each
        # header_info starts with the signed offset from itself to its image's mach header.
        table = cache.region + header_info_ro
        count, entsize = struct.unpack('<2I', cache.read(table, 8))
        require(entsize >= 16 and 0 < count < 1 << 16, 'header_info table shape')
        self.header_images = []
        for i in range(count):
            entry = table + 8 + i * entsize
            mhdr_offset, = struct.unpack('<q', cache.read(entry, 8))
            require(struct.unpack('<I', cache.read(entry + mhdr_offset, 4))[0] == 0xfeedfacf,
                    f'header_info {i} does not reach a mach header')
            self.header_images.append(entry + mhdr_offset)
        self._classes = {}
        self._class_names = {}

    # -- lists ------------------------------------------------------------------------------
    def method_list(self, address, image_index=None):
        if not address:
            return []
        cache = self.cache
        flags, count = struct.unpack('<2I', cache.read(address, 8))
        entsize = flags & ~METHOD_LIST_FLAGS & 0xffff
        small = bool(flags & SMALL_METHOD_LIST)
        require(count <= LIST_BOUND and entsize == (12 if small else 24),
                f'method list {address:#x}: flags {flags:#x} count {count}')
        out = []
        for i in range(count):
            entry = address + 8 + i * entsize
            if small:
                name_off, types_off, imp_off = struct.unpack('<3i', cache.read(entry, 12))
                if flags & RELATIVE_SELECTORS_DIRECT:
                    name = self.selector_base + name_off
                else:
                    name = cache.pointer(entry + name_off)
                types = entry + 4 + types_off
                imp = entry + 8 + imp_off if imp_off else 0
            else:
                name, types, imp = cache.pointer(entry), cache.pointer(entry + 8), cache.pointer(entry + 16)
            selector, encoding = cache.cstring(name), cache.cstring(types)
            require(selector and imp % 4 == 0, f'method list {address:#x} entry {i}')
            out.append(Method(selector, encoding, imp, image_index))
        return out

    def method_lists(self, address):
        """A class's method lists in lookup order; handles shared-cache list-of-lists."""
        if not address:
            return []
        if not address & 1:
            return [self.method_list(address)]
        header = address & ~1
        entsize, count = struct.unpack('<2I', self.cache.read(header, 8))
        require(entsize == 8 and 0 < count <= 4096, f'relative list-of-lists {header:#x}')
        lists = []
        for i in range(count):
            entry = header + 8 + 8 * i
            raw, = struct.unpack('<q', self.cache.read(entry, 8))
            image_index, offset = raw & 0xffff, raw >> 16
            lists.append(self.method_list(entry + offset, image_index))
        return lists

    def ivar_list(self, address):
        if not address:
            return []
        cache = self.cache
        entsize, count = struct.unpack('<2I', cache.read(address, 8))
        require(entsize == 32 and count <= LIST_BOUND, f'ivar list {address:#x}')
        out = []
        for i in range(count):
            entry = address + 8 + 32 * i
            offset_ptr = cache.pointer(entry)
            name = cache.cstring(cache.pointer(entry + 8))
            type_ = cache.cstring(cache.pointer(entry + 16), 1 << 16)
            alignment_raw, size = struct.unpack('<2I', cache.read(entry + 24, 8))
            offset = struct.unpack('<I', cache.read(offset_ptr, 4))[0] if offset_ptr else None
            out.append(Ivar(name, type_, offset, size, 1 << alignment_raw if alignment_raw != 0xffffffff else 8))
        return out

    def protocol_list(self, address):
        if not address:
            return []
        if address & 1:
            out = []
            header = address & ~1
            entsize, count = struct.unpack('<2I', self.cache.read(header, 8))
            require(entsize == 8 and 0 < count <= 4096, f'relative protocol list-of-lists {header:#x}')
            for i in range(count):
                entry = header + 8 + 8 * i
                raw, = struct.unpack('<q', self.cache.read(entry, 8))
                out.extend(self.protocol_list(entry + (raw >> 16)))
            return out
        count, = struct.unpack('<Q', self.cache.read(address, 8))
        require(count <= 4096, f'protocol list {address:#x}')
        return [self.cache.pointer(address + 8 + 8 * i) for i in range(count)]

    def protocol_methods(self, address):
        if not address:
            return []
        cache = self.cache
        flags, count = struct.unpack('<2I', cache.read(address, 8))
        entsize = flags & ~METHOD_LIST_FLAGS & 0xffff
        require(count <= LIST_BOUND, f'protocol method list {address:#x}')
        out = []
        for i in range(count):
            entry = address + 8 + i * entsize
            if flags & SMALL_METHOD_LIST:
                name_off, types_off, _ = struct.unpack('<3i', cache.read(entry, 12))
                name = self.selector_base + name_off if flags & RELATIVE_SELECTORS_DIRECT else cache.pointer(entry + name_off)
                types = entry + 4 + types_off
            else:
                require(entsize == 24, f'protocol method list {address:#x} entsize {entsize}')
                name, types = cache.pointer(entry), cache.pointer(entry + 8)
            out.append((cache.cstring(name), cache.cstring(types)))
        return out

    # -- classes ----------------------------------------------------------------------------
    def klass(self, address):
        if address not in self._classes:
            self._classes[address] = Class(self, address)
        return self._classes[address]

    def image_classes(self, image):
        out = {}
        for _, start, size in image.section('__DATA', '__objc_classlist'):
            for at in range(start, start + size, 8):
                address = self.cache.pointer(at)
                out[self.klass(address).name] = address
        return out

    def class_named(self, image, name):
        key = (image.path, name)
        if key not in self._class_names:
            classes = self.image_classes(image)
            require(name in classes, f'{image.name}: no class {name}')
            self._class_names[key] = classes[name]
        return self.klass(self._class_names[key])

    def protocol_named(self, image, name):
        for _, start, size in image.section('__DATA', '__objc_protolist'):
            for at in range(start, start + size, 8):
                address = self.cache.pointer(at)
                if self.cache.cstring(self.cache.pointer(address + 8)) == name:
                    return Protocol(self, address)
        raise CacheError(f'{image.name}: no protocol {name}')

    def protocol_closure(self, protocol):
        """The protocol and every protocol it adopts, depth first, each once."""
        seen, order = set(), []

        def visit(address):
            if address in seen:
                return
            seen.add(address)
            p = Protocol(self, address)
            order.append(p)
            for inner in p.protocols:
                visit(inner)
        visit(protocol.address)
        return order

    def chain(self, klass):
        """The class and its superclasses, nearest first."""
        out = []
        while klass is not None:
            require(len(out) < 32, 'class chain bound')
            out.append(klass)
            klass = self.klass(klass.superclass) if klass.superclass else None
        return out

    def lookup(self, klass, selector):
        """(defining class, Method) for selector, searching each class's lists in order."""
        for c in self.chain(klass):
            for methods in c.method_lists:
                for m in methods:
                    if m.selector == selector:
                        return c, m
        return None, None

    def ivar(self, klass, name):
        for c in self.chain(klass):
            for ivar in c.ivars:
                if ivar.name == name:
                    return c, ivar
        raise CacheError(f'{klass.name}: no ivar {name}')

    def ivar_at(self, klass, offset):
        """(class, ivar) whose storage contains the byte at offset."""
        for c in self.chain(klass):
            for ivar in c.ivars:
                if ivar.offset is not None and ivar.offset <= offset < ivar.offset + max(ivar.size, 1):
                    return c, ivar
        return None, None

    def categories_on(self, class_addresses):
        """Scan every image's category list for categories whose class is in class_addresses."""
        found = []
        cache = self.cache
        for image in cache.images:
            for section in ('__objc_catlist', '__objc_catlist2'):
                for _, start, size in image.section('__DATA', section):
                    for at in range(start, start + size, 8):
                        cat = cache.pointer(at)
                        cls = cache.pointer(cat + 8)
                        if cls in class_addresses:
                            name = cache.cstring(cache.pointer(cat))
                            methods = self.method_list(cache.pointer(cat + 16))
                            found.append((image.path, name, cls, methods))
        return found
