#!/usr/bin/env python3
"""Derive the Metal driver surface of the iPadOS 17.7.11 (21H461) target and check the wire schema against it.

Reads the target's unchanged dyld shared cache read-only (mmap, no side files) and decodes the Objective-C metadata of the
images that matter: Metal.framework (the `_MTL*` bases and every `MTL*` protocol), IOGPU.framework, the AGX driver bundle
`AGXMetalA10` and QuartzCore. Nothing it reads is written out except selector names, protocol names and labels derived here.

Commands:

  derive  Print the QuartzCore driver-selector list (the selectors QuartzCore references that a Metal driver object
          implements), one line per selector: selector, the driver protocols that declare it, the layer that implements it
          in the target's driver stack, and its category. `--write` replaces `wire/target/quartzcore-driver-selectors.tsv`.
  check   Check `wire/schema/webmetal.toml` against the binaries: every op's argument list against the declaring protocol's
          extended method type encoding, every descriptor struct against its class's accessors, every guest-local annotation
          against the protocol and implementation layer, the committed selector list against a fresh derivation, and coverage
          of the driver protocols. Exits 1 on any error. Prints the unclassified backlog per protocol.

Definitions (they make the list reproducible):

  driver classes      the classes in AGXMetalA10's __objc_classlist.
  driver chain        those classes and their superclasses in IOGPU.framework and Metal.framework.
  driver protocols    the MTL* protocols adopted anywhere in the driver chain, closed over protocol inheritance.
  driver selectors    selectors declared by a driver protocol, or implemented by a driver class.
  QuartzCore list     QuartzCore's __objc_selrefs that are driver selectors or are declared by any other MTL* protocol in the
                      cache, minus runtime-generated and root-class selectors (.cxx_construct, .cxx_destruct, dealloc, init,
                      initialize). An entry that no driver protocol declares and that the AGX and IOGPU layers do not
                      implement is category `framework-object`; every other entry is category `driver`.

A selector reference proves QuartzCore can send that message, not to which receiver; where several protocols declare a
selector, all of them are listed.
"""
import argparse
import mmap
import re
import struct
import sys
import tomllib
from pathlib import Path

CACHE_REL = 'System/Library/Caches/com.apple.dyld/dyld_shared_cache_arm64'
CACHE_UUID = '590d839a82d230f1a17d0cc053420f73'
IMAGES = {
    'Metal': '/System/Library/Frameworks/Metal.framework/Metal',
    'IOGPU': '/System/Library/PrivateFrameworks/IOGPU.framework/IOGPU',
    'AGX': '/System/Library/Extensions/AGXMetalA10.bundle/AGXMetalA10',
    'QuartzCore': '/System/Library/Frameworks/QuartzCore.framework/QuartzCore',
}
EXCLUDED_SELECTORS = {'.cxx_construct', '.cxx_destruct', 'dealloc', 'init', 'initialize'}
WIRE = Path(__file__).resolve().parents[1]
LIST_PATH = WIRE / 'target/quartzcore-driver-selectors.tsv'
SCHEMA_PATH = WIRE / 'schema/webmetal.toml'
ACCEPTANCE = re.compile(r'^(MTL\w*CommandEncoder(SPI)?|MTLDevice|MTLDeviceSPI)$')


class Cache:
    """dyld_shared_cache_arm64 with its sub-caches. Header layout: dyld's dyld_cache_format.h (dyld-1042 era)."""

    def __init__(self, path):
        self.files = []
        self.maps = []
        self.slides = []
        main = self._open(path)
        if main[0x58:0x68].hex() != CACHE_UUID:
            raise SystemExit('not the 21H461 cache: UUID ' + main[0x58:0x68].hex())
        self.main = main
        mapping_offset, = struct.unpack_from('<I', main, 0x10)
        self.base, = struct.unpack_from('<Q', main, mapping_offset)
        sub_offset, sub_count = struct.unpack_from('<II', main, 0x188)
        self._add(main)
        for i in range(sub_count):
            entry = main[sub_offset + 56 * i:sub_offset + 56 * (i + 1)]
            self._add(self._open(path + entry[24:56].split(b'\0')[0].decode()))
        objc_offset, = struct.unpack_from('<Q', main, 0x1d0)
        header = struct.unpack('<II6Q', self.read(self.base + objc_offset, 56))
        self.selector_base = self.base + header[7]

    def _open(self, path):
        stream = open(path, 'rb')
        data = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.files.append((stream, data))
        return data

    def _add(self, data):
        if data[:16].rstrip(b'\0') != b'dyld_v1   arm64':
            raise SystemExit('not an arm64 dyld cache file')
        at, count = struct.unpack_from('<II', data, 0x138)
        for i in range(count):
            va, size, offset, slide_offset, slide_size, _, _, _ = struct.unpack_from('<6Q2I', data, at + 56 * i)
            self.maps.append((va, va + size, offset, data))
            if slide_size:
                version, = struct.unpack_from('<I', data, slide_offset)
                if version != 2:
                    raise SystemExit('unsupported slide info version %d' % version)
                mask, add = struct.unpack_from('<2Q', data, slide_offset + 24)
                self.slides.append((va, va + size, mask, add))

    def read(self, va, size):
        for start, end, offset, data in self.maps:
            if start <= va and va + size <= end:
                return data[offset + va - start:offset + va - start + size]
        raise SystemExit('unbacked cache address 0x%x' % va)

    def pointer(self, va):
        raw, = struct.unpack('<Q', self.read(va, 8))
        for start, end, mask, add in self.slides:
            if start <= va < end:
                value = raw & ~mask & 0xffffffffffffffff
                return value + add if value else 0
        return raw

    def cstring(self, va):
        for start, end, offset, data in self.maps:
            if start <= va < end:
                at = offset + va - start
                stop = data.find(b'\0', at, offset + end - start)
                return data[at:stop].decode('utf-8', 'replace')
        raise SystemExit('unbacked cache string 0x%x' % va)

    def images(self):
        offset, count = struct.unpack_from('<II', self.main, 0x1c0)
        out = {}
        for i in range(count):
            address, _, _, path_offset, _ = struct.unpack_from('<3Q2I', self.main, offset + 32 * i)
            out[self.main[path_offset:self.main.find(b'\0', path_offset)].decode()] = address
        return out

    def sections(self, address):
        magic, _, _, _, ncmds, size, _, _ = struct.unpack('<8I', self.read(address, 32))
        if magic != 0xfeedfacf:
            raise SystemExit('not a 64-bit Mach-O at 0x%x' % address)
        commands = self.read(address + 32, size)
        at, out = 0, {}
        for _ in range(ncmds):
            cmd, cmdsize = struct.unpack_from('<II', commands, at)
            if cmd == 0x19:  # LC_SEGMENT_64
                segment = commands[at + 8:at + 24].split(b'\0')[0].decode()
                nsects, = struct.unpack_from('<I', commands, at + 64)
                for k in range(nsects):
                    s = at + 72 + 80 * k
                    va, sz = struct.unpack_from('<2Q', commands, s + 32)
                    out[(segment, commands[s:s + 16].split(b'\0')[0].decode())] = (va, sz)
            at += cmdsize
        return out


class ObjC:
    """Objective-C runtime structures as objc4 lays them out for arm64 shared-cache images (objc-runtime-new.h)."""

    SMALL = 0x80000000          # method_t::smallMethodListFlag
    DIRECT_SELECTORS = 0x40000000  # selectors are offsets from the cache's relative method selector base

    def __init__(self, cache):
        self.c = cache

    def lists(self, field):
        """A list field: a plain list, or with the low bit set a relative_list_list_t of lists."""
        if not field:
            return []
        if not field & 1:
            return [field]
        field &= ~1
        entsize, count = struct.unpack('<II', self.c.read(field, 8))
        out = []
        for i in range(count):
            entry = field + 8 + entsize * i
            raw, = struct.unpack('<Q', self.c.read(entry, 8))
            offset = raw >> 16
            if offset & (1 << 47):
                offset -= 1 << 48
            out.append(entry + offset)
        return out

    def methods(self, field):
        out = []
        for address in self.lists(field):
            flags, count = struct.unpack('<II', self.c.read(address, 8))
            if flags & self.SMALL:
                if not flags & self.DIRECT_SELECTORS:
                    raise SystemExit('relative method list without direct selectors at 0x%x' % address)
                for i in range(count):
                    entry = address + 8 + 12 * i
                    name, types, _ = struct.unpack('<3i', self.c.read(entry, 12))
                    out.append((self.c.cstring(self.c.selector_base + name), self.c.cstring(entry + 4 + types)))
            else:
                entsize = flags & 0xfffc
                for i in range(count):
                    entry = address + 8 + entsize * i
                    out.append((self.c.cstring(self.c.pointer(entry)), self.c.cstring(self.c.pointer(entry + 8))))
        return out

    def properties(self, field):
        out = []
        for address in self.lists(field):
            entsize, count = struct.unpack('<II', self.c.read(address, 8))
            for i in range(count):
                entry = address + 8 + entsize * i
                out.append((self.c.cstring(self.c.pointer(entry)), self.c.cstring(self.c.pointer(entry + 8))))
        return out

    def protocol_list(self, field):
        out = []
        for address in self.lists(field):
            count, = struct.unpack('<Q', self.c.read(address, 8))
            out += [self.c.pointer(address + 8 + 8 * i) for i in range(count)]
        return out

    def protocol_name(self, address):
        return self.c.cstring(self.c.pointer(address + 8))

    def protocol(self, address):
        fields = [self.c.pointer(address + 8 * i) for i in range(8)]
        size, _ = struct.unpack('<II', self.c.read(address + 64, 8))
        extended = self.c.pointer(address + 72) if size >= 80 else 0
        methods, k = [], 0
        for kind, field in (('required', fields[3]), ('required-class', fields[4]),
                            ('optional', fields[5]), ('optional-class', fields[6])):
            for name, types in self.methods(field):
                ext = self.c.cstring(self.c.pointer(extended + 8 * k)) if extended else types
                methods.append({'kind': kind, 'name': name, 'types': types, 'ext': ext})
                k += 1
        return {'name': self.c.cstring(fields[1]), 'protocols': [self.protocol_name(p) for p in self.protocol_list(fields[2])],
                'methods': methods, 'properties': self.properties(fields[7])}

    def data(self, cls):
        return self.c.pointer(cls + 32) & 0x00007ffffffffff8  # FAST_DATA_MASK

    def class_name(self, cls):
        return self.c.cstring(self.c.pointer(self.data(cls) + 24))

    def objc_class(self, cls):
        ro = self.data(cls)
        superclass = self.c.pointer(cls + 8)
        meta = self.data(self.c.pointer(cls))
        return {'name': self.c.cstring(self.c.pointer(ro + 24)),
                'super': self.class_name(superclass) if superclass else None,
                'methods': self.methods(self.c.pointer(ro + 32)),
                'class_methods': self.methods(self.c.pointer(meta + 32)),
                'protocols': [self.protocol_name(p) for p in self.protocol_list(self.c.pointer(ro + 40))],
                'properties': self.properties(self.c.pointer(ro + 64))}

    def section(self, sections, name):
        for segment in ('__DATA_CONST', '__DATA', '__DATA_DIRTY', '__AUTH_CONST', '__AUTH'):
            if (segment, name) in sections:
                yield sections[(segment, name)]

    def image(self, address):
        sections = self.c.sections(address)
        out = {'classes': [], 'selrefs': set(), 'protocols': []}
        for va, size in self.section(sections, '__objc_classlist'):
            out['classes'] += [self.objc_class(self.c.pointer(at)) for at in range(va, va + size, 8)]
        for va, size in self.section(sections, '__objc_selrefs'):
            for at in range(va, va + size, 8):
                target = self.c.pointer(at)
                if target:
                    out['selrefs'].add(self.c.cstring(target))
        for va, size in self.section(sections, '__objc_protolist'):
            out['protocols'] += [self.c.pointer(at) for at in range(va, va + size, 8)]
        return out


class Target:
    def __init__(self, root):
        self.cache = Cache(str(Path(root) / CACHE_REL))
        self.objc = ObjC(self.cache)
        images = self.cache.images()
        self.images = {key: self.objc.image(images[path]) for key, path in IMAGES.items()}
        self.classes = {}
        for key in ('Metal', 'IOGPU', 'AGX', 'QuartzCore'):
            for cls in self.images[key]['classes']:
                cls['image'] = key
                self.classes[cls['name']] = cls
        self.protocols = self._protocols(images)

    def _protocols(self, images):
        """Every MTL* protocol definition in the cache. Copies in different images differ only in how dispatch types are
        spelled (`NSObject<OS_dispatch_queue>` or `OS_dispatch_queue`); Metal's own copy is preferred, then the lowest path."""
        found = {}
        for path in sorted(images):
            sections = self.cache.sections(images[path])
            for va, size in self.objc.section(sections, '__objc_protolist'):
                for at in range(va, va + size, 8):
                    address = self.cache.pointer(at)
                    name = self.objc.protocol_name(address)
                    if not name.startswith('MTL') or name in found and found[name][0] == IMAGES['Metal']:
                        continue
                    if name not in found or path == IMAGES['Metal']:
                        found[name] = (path, self.objc.protocol(address))
        return {name: definition for name, (_, definition) in found.items()}

    def closure(self, names):
        out, stack = set(), list(names)
        while stack:
            name = stack.pop()
            if name in out:
                continue
            out.add(name)
            if name in self.protocols:
                stack += self.protocols[name]['protocols']
        return out

    def chain(self, name):
        out = []
        while name in self.classes:
            out.append(self.classes[name])
            name = self.classes[name]['super']
        return out

    def driver_protocols(self):
        adopted = set()
        for cls in self.images['AGX']['classes']:
            for link in self.chain(cls['name']):
                adopted |= set(link['protocols'])
        return {p for p in self.closure(adopted) if p.startswith('MTL') and p in self.protocols}

    def conforms(self, class_name, protocol):
        """Does the class adopt the protocol, or a protocol the given one extends (an object conforming to MTLBuffer
        receives MTLBufferSPI messages)?"""
        adopted = set()
        for link in self.chain(class_name):
            adopted |= set(link['protocols'])
        family = self.closure([protocol]) - {'NSObject'}
        return bool(self.closure(adopted) & family)

    def driver_classes(self):
        """The classes a replacement plugin replaces: the AGX bundle's and IOGPU.framework's. The plugin subclasses
        Metal.framework's _MTL* bases directly and uses neither."""
        return self.images['AGX']['classes'] + self.images['IOGPU']['classes']

    def implementation_layer(self, selector, protocol=None):
        """Where the target's driver stack implements a selector: 'agx' (the driver bundle), 'iogpu' (IOGPU.framework),
        'framework' (a Metal.framework base class of a driver class), or 'none'. With a protocol, only driver classes
        that conform to it (or to a protocol it extends) count."""
        layers = set()
        for cls in self.driver_classes():
            if protocol and not self.conforms(cls['name'], protocol):
                continue
            for link in self.chain(cls['name']):
                if any(m[0] == selector for m in link['methods'] + link['class_methods']):
                    layers.add({'AGX': 'agx', 'IOGPU': 'iogpu', 'Metal': 'framework'}[link['image']])
                    break
        for layer in ('agx', 'iogpu', 'framework'):
            if layer in layers:
                return layer
        return 'none'

    def quartzcore_list(self):
        driver = self.driver_protocols()
        declared = {}
        for name, proto in self.protocols.items():
            for m in proto['methods']:
                declared.setdefault(m['name'], set()).add(name)
        implemented = {m[0] for cls in self.images['AGX']['classes'] for m in cls['methods'] + cls['class_methods']}
        rows = []
        for selector in sorted((self.images['QuartzCore']['selrefs'] & (set(declared) | implemented)) - EXCLUDED_SELECTORS):
            protocols = sorted(declared.get(selector, ()))
            in_driver = sorted(p for p in protocols if p in driver)
            layer = self.implementation_layer(selector)
            if in_driver or layer in ('agx', 'iogpu'):
                rows.append((selector, in_driver or protocols, layer, 'driver'))
            else:
                rows.append((selector, protocols, layer, 'framework-object'))
        return rows


def list_text(rows):
    lines = ['# selector\tprotocols\timplementation-layer\tcategory',
             '# Generated by wire/tools/target_surface.py derive --write from the 21H461 dyld shared cache.']
    lines += ['%s\t%s\t%s\t%s' % (sel, ','.join(protos), layer, category) for sel, protos, layer, category in rows]
    return '\n'.join(lines) + '\n'


# Objective-C type encodings -------------------------------------------------------------------------------------------

def parse_type(text, at):
    """Parse one Objective-C (extended) type encoding starting at `at`; return (encoding, next index)."""
    start = at
    while text[at] in 'rnNoORV':  # qualifiers: const, in, inout, out, bycopy, byref, oneway
        at += 1
    ch = text[at]
    if ch == '@':
        at += 1
        if at < len(text) and text[at] == '?':
            at += 1
            if at < len(text) and text[at] == '<':
                depth = 0
                while True:
                    if text[at] == '<':
                        depth += 1
                    elif text[at] == '>':
                        depth -= 1
                        if not depth:
                            at += 1
                            break
                    at += 1
        elif at < len(text) and text[at] == '"':
            at = text.index('"', at + 1) + 1
    elif ch == '^':
        _, at = parse_type(text, at + 1)
    elif ch in '{(':
        close = '}' if ch == '{' else ')'
        depth = 0
        while True:
            if text[at] in '{(':
                depth += 1
            elif text[at] in '})':
                depth -= 1
                if not depth:
                    at += 1
                    break
            elif text[at] == '"':
                at = text.index('"', at + 1)
            at += 1
        del close
    elif ch == '[':
        at += 1
        while text[at].isdigit():
            at += 1
        _, at = parse_type(text, at)
        at += 1
    elif ch == 'b':
        at += 1
        while at < len(text) and text[at].isdigit():
            at += 1
    else:
        at += 1
    return text[start:at], at


def method_arguments(encoding):
    """Return (return type, [argument types]) of a method encoding, without self and _cmd."""
    types, at = [], 0
    while at < len(encoding):
        t, at = parse_type(encoding, at)
        while at < len(encoding) and encoding[at].isdigit():
            at += 1
        types.append(t)
    if len(types) < 3 or types[1] != '@' or types[2] != ':':
        raise ValueError('not a method encoding: ' + encoding)
    return types[0], types[3:]


def strip_names(encoding):
    """Drop extended-encoding class names and block signatures so that two encodings compare by shape."""
    encoding = re.sub(r'@"[^"]*"', '@', encoding)
    encoding = re.sub(r'@\?<[^>]*(<[^>]*>[^>]*)*>', '@?', encoding)
    return encoding


# Schema --------------------------------------------------------------------------------------------------------------

SCALARS = {'u8': 'C', 'u16': 'S', 'u32': 'I', 'u64': 'Q', 'i32': 'i', 'i64': 'q', 'f32': 'f', 'f64': 'd', 'bool': 'B'}
LOCAL_REASONS = {
    'profile': 'answered from the A10X device profile; no host state is involved',
    'object-state': 'a value the guest object holds: given at creation, set by a setter whose value travels in another record, or derived from both',
    'framework': 'implemented by Metal.framework or QuartzCore code that the plugin inherits or does not replace; the 21H461 driver does not implement it',
    'guest-runtime': 'guest-side machinery (handlers, waits, dispatch queues, timing fields) driven by the event records',
    'guest-memory': 'reads or writes only guest memory in a layout the plugin defines and the wire documents',
    'kernel-object': 'served by another framework against a kernel object (IOSurface shared events); the host reaches the same object through the embedder',
    'unreachable': 'no layer of the 21H461 driver hands out an object of this protocol: the optional factory method named in the annotation is itself absent',
    'unavailable': 'declared by the protocol but unavailable on iOS (API_UNAVAILABLE(ios) in the SDK) and implemented by no layer of the 21H461 driver',
    'absent': 'an optional protocol method that no layer of the 21H461 driver implements; the plugin does not implement it either, so callers that check respondsToSelector: behave as on the device',
}
FIELD = re.compile(r'^(\w+): (.+?)(?: \((\w+)\))?(?: = (\w+(?:\.\w+)?))?$')


def parse_field(text):
    """'name: type (MetalType) = constraint' -> (name, type, MetalType or None, constraint or None)."""
    m = FIELD.match(text)
    if not m:
        raise SystemExit('malformed field: ' + text)
    return m.group(1), m.group(2), m.group(3), m.group(4)


class Checker:
    def __init__(self, schema, target):
        self.schema = schema
        self.target = target
        self.errors = []
        self.structs = {s['name']: s for s in schema.get('struct', [])}
        self.objects = {o['name']: o for o in schema.get('object', [])}

    def error(self, message):
        self.errors.append(message)

    def element_encoding_ok(self, wire, encoding, where):
        """Does a wire element type correspond to an Objective-C encoding (array element or pointer target)?"""
        wire = wire.rstrip('?')
        if wire in SCALARS:
            return encoding == SCALARS[wire]
        if wire in self.objects:
            return encoding == '@' or self.object_ok(wire, encoding)
        if wire in self.structs:
            return strip_names(self.structs[wire].get('encoding', '')) == strip_names(encoding)
        return False

    def object_ok(self, kind, encoding):
        m = re.match(r'^@"<(\w+)>"$', encoding)
        if not m:
            return encoding == '@'
        protocols = self.objects[kind].get('protocols', [])
        return m.group(1) in protocols

    def type_ok(self, wire, encoding):
        """Does a schema wire type carry a Metal argument with this extended encoding?"""
        bare = re.sub(r'^[rnNoORV]+', '', encoding)
        if wire in SCALARS:
            return bare == SCALARS[wire]
        if wire.endswith('?') and wire[:-1] in self.objects:
            return self.object_ok(wire[:-1], bare)
        if wire in self.objects:
            return self.object_ok(wire, bare)
        if wire.rstrip('?') in self.structs:
            s = self.structs[wire.rstrip('?')]
            if 'class' in s:
                m = re.match(r'^@"(\w+)"$', bare)
                return bool(m) and m.group(1) in s.get('classes', [s['class'].removesuffix('Internal')])
            return strip_names(s.get('encoding', '')) == strip_names(bare)
        if wire == 'str?' or wire == 'str':
            return bare in ('@"NSString"', '@')
        if wire == 'url?':
            return bare in ('@"NSURL"', '@')
        if wire == 'cstr?':
            return bare == '*'
        if wire == 'value?' or wire == 'value':
            return bare in ('@"NSDictionary"', '@"NSArray"', '@')
        if wire == 'block':
            return bare.startswith('@?')
        if wire == 'out':
            return bare == '^@'
        if wire == 'present':
            return bare.startswith('@') and not bare.startswith('@?')
        if wire == 'surface':
            return bare == '^{__IOSurface=}'
        if wire == 'guest_addr':
            return bare in ('^v', '^Q')
        if wire == 'bytes':
            return bare in ('^v', '*')
        m = re.match(r'^\[(\w+\??)\]$', wire)
        if m and bare == '@"NSDictionary"':
            return self.structs.get(m.group(1), {}).get('dictionary_entry', False)
        if m:
            if not bare.startswith('^'):
                return False
            return self.element_encoding_ok(m.group(1), re.sub(r'^\^[rnNoORV]*', '', bare), wire)
        return False

    def check_ops(self):
        seen_codes = {}
        claimed = {}
        for op in self.schema.get('op', []) + self.schema.get('event', []):
            code = op['code']
            if code in seen_codes:
                self.error('opcode 0x%04x used by %s and %s' % (code, seen_codes[code], op['name']))
            seen_codes[code] = op['name']
            fields = [parse_field(a) for a in op.get('args', [])]
            metal_fields = [f for f in fields if f[0] not in op.get('extra', []) and f[0] != op.get('receiver')
                            and f[0] != op.get('result')]
            for method in op.get('methods', []):
                protocol, selector = method.split(' ')
                claimed.setdefault((protocol, selector), []).append(op['name'])
                proto = self.target.protocols.get(protocol)
                if not proto:
                    self.error('%s: unknown protocol %s' % (op['name'], protocol))
                    continue
                matches = [m for m in proto['methods'] if m['name'] == selector]
                if not matches:
                    self.error('%s: %s does not declare %s on 21H461' % (op['name'], protocol, selector))
                    continue
                layer = self.target.implementation_layer(selector, protocol)
                if layer not in ('agx', 'iogpu') and not op.get('plugin_override'):
                    self.error('%s: %s %s is not implemented by the 21H461 driver (%s layer); mark plugin_override with '
                               'the reason or annotate it guest-local' % (op['name'], protocol, selector, layer))
                _, arguments = method_arguments(matches[0]['ext'])
                if len(arguments) != len(metal_fields):
                    self.error('%s: %s %s takes %d arguments, schema maps %d' % (
                        op['name'], protocol, selector, len(arguments), len(metal_fields)))
                    continue
                for (name, wire, _, _), encoding in zip(metal_fields, arguments):
                    if not self.type_ok(wire, encoding):
                        self.error('%s: %s %s argument %s: wire type %s does not carry %s' % (
                            op['name'], protocol, selector, name, wire, encoding))
        return claimed

    def accessors(self, class_name):
        """Settable state of a descriptor class: getter/setter pairs of one argument, plus readonly collection properties."""
        methods = {}
        properties = {}
        for link in reversed(self.target.chain(class_name)):
            for name, types in link['methods']:
                methods[name] = types
            for name, attributes in link['properties']:
                properties[name] = attributes
        out = {}
        for name, types in methods.items():
            if name.startswith('set') and name.endswith(':') and name.count(':') == 1:
                base = name[3:-1]
                field = base[0].lower() + base[1:]
                for getter in (field, 'is' + base, base):
                    if getter in methods:
                        _, args = method_arguments(types)
                        out[field] = args[0]
                        break
        for name, attributes in properties.items():
            parts = attributes.split(',')
            if 'R' in parts and name not in out:
                m = re.match(r'^T(@"\w+")$', parts[0])
                if m and m.group(1).endswith('Array"'):
                    out[name] = m.group(1)
        return out, properties

    def check_structs(self):
        for s in self.schema.get('struct', []):
            if 'class' not in s:
                continue
            if s['class'] not in self.target.classes:
                self.error('struct %s: class %s not in Metal on 21H461' % (s['name'], s['class']))
                continue
            state, properties = self.accessors(s['class'])
            fields = {parse_field(f)[0]: parse_field(f) for f in s.get('fields', [])}
            covered = set()
            for field_name, (_, wire, _, _) in fields.items():
                source = s.get('source', {}).get(field_name, field_name_to_accessor(field_name))
                if source in s.get('accessors', {}):
                    covered.add(source)
                    continue
                if source not in state:
                    self.error('struct %s: field %s has no accessor %s on %s' % (s['name'], field_name, source, s['class']))
                    continue
                covered.add(source)
                encoding = state[source]
                prop = properties.get(source, '')
                m = re.match(r'^T(@"[^"]+")', prop)
                if encoding == '@' and m:
                    encoding = m.group(1)
                if encoding.endswith('Array"'):
                    continue  # collections map to arrays of element structs, checked through `element`
                if not self.type_ok(wire, encoding) and not (encoding == '@' and self.object_field_ok(wire)):
                    self.error('struct %s: field %s wire type %s does not carry %s' % (s['name'], field_name, wire, encoding))
            for accessor in s.get('accessors', {}):
                if not any(m[0] == accessor for link in self.target.chain(s['class']) for m in link['methods']):
                    self.error('struct %s: accessor %s not on %s' % (s['name'], accessor, s['class']))
            excluded = s.get('excluded', {})
            for accessor in state:
                if accessor not in covered and accessor not in excluded:
                    self.error('struct %s: %s state %s has no field and is not excluded' % (s['name'], s['class'], accessor))
            for accessor in excluded:
                if accessor not in state:
                    self.error('struct %s: excluded %s is not state of %s' % (s['name'], accessor, s['class']))

    def object_field_ok(self, wire):
        base = wire.rstrip('?')
        return (base in self.objects or base in self.structs or wire in ('str?', 'url?', 'value?', 'present')
                or re.match(r'^\[\w+\??\]$', wire) is not None)

    def check_locals(self, claimed):
        local = {}
        for entry in self.schema.get('local', []):
            protocols = entry['protocols']
            for selector in entry['selectors']:
                targets = protocols if protocols != ['*'] else [p for p, d in self.target.protocols.items()
                                                                 if any(m['name'] == selector for m in d['methods'])]
                if not targets:
                    self.error('local %s: no protocol declares it' % selector)
                for protocol in targets:
                    proto = self.target.protocols.get(protocol)
                    if not proto or not any(m['name'] == selector for m in proto['methods']):
                        self.error('local: %s does not declare %s' % (protocol, selector))
                        continue
                    if (protocol, selector) in claimed:
                        self.error('%s %s is both local and op %s' % (protocol, selector, claimed[(protocol, selector)]))
                    local[(protocol, selector)] = entry['reason']
                if entry['reason'] not in LOCAL_REASONS:
                    self.error('local %s: unknown reason %s' % (selector, entry['reason']))
                if entry['reason'] == 'unreachable':
                    factory = entry.get('factory', '')
                    fp, _, fs = factory.partition(' ')
                    proto = self.target.protocols.get(fp, {'methods': []})
                    kinds = {m['kind'] for m in proto['methods'] if m['name'] == fs}
                    if not kinds or not all(k.startswith('optional') for k in kinds) or \
                            self.target.implementation_layer(fs, fp) != 'none':
                        self.error('local %s marked unreachable but its factory %r is not an absent optional method' % (
                            selector, factory))
                if entry['reason'] == 'unavailable':
                    for protocol in targets:
                        layer = self.target.implementation_layer(selector, protocol)
                        if layer != 'none':
                            self.error('local %s %s marked unavailable but the %s layer implements it' % (protocol, selector, layer))
                if entry['reason'] == 'absent':
                    for protocol in targets:
                        proto = self.target.protocols.get(protocol, {'methods': []})
                        kinds = {m['kind'] for m in proto['methods'] if m['name'] == selector}
                        if not kinds or not all(k.startswith('optional') for k in kinds):
                            self.error('local %s %s marked absent but the protocol requires it' % (protocol, selector))
                        layer = self.target.implementation_layer(selector, protocol)
                        if layer != 'none':
                            self.error('local %s %s marked absent but the %s layer implements it' % (protocol, selector, layer))
                if entry['reason'] == 'framework':
                    for protocol in targets:
                        layer = self.target.implementation_layer(selector, protocol)
                        if layer in ('agx', 'iogpu'):
                            self.error('local %s %s marked framework but the driver implements it (%s layer)' % (
                                protocol, selector, layer))
        return local

    def check_list(self):
        fresh = list_text(self.target.quartzcore_list())
        if not LIST_PATH.exists() or LIST_PATH.read_text() != fresh:
            self.error('%s differs from a fresh derivation; run derive --write' % LIST_PATH.relative_to(WIRE.parent))

    def coverage(self, claimed, local):
        driver = self.target.driver_protocols()
        rows = self.target.quartzcore_list()
        backlog = {}
        for protocol in sorted(driver):
            for m in self.target.protocols[protocol]['methods']:
                if (protocol, m['name']) not in claimed and (protocol, m['name']) not in local:
                    backlog.setdefault(protocol, []).append(m['name'])
        qc_pairs = []
        for selector, protocols, _, category in rows:
            for protocol in protocols:
                qc_pairs.append((protocol, selector))
        missing = [(p, s) for p, s in qc_pairs if (p, s) not in claimed and (p, s) not in local]
        for protocol, selector in missing:
            if ACCEPTANCE.match(protocol):
                self.error('QuartzCore selector %s %s (MTL*CommandEncoder/MTLDevice/MTLDeviceSPI) has neither an opcode '
                           'nor a guest-local annotation' % (protocol, selector))
        acceptance = [(p, s) for p, s in qc_pairs if ACCEPTANCE.match(p)]
        self.quartzcore_backlog = [(p, s) for p, s in missing if not ACCEPTANCE.match(p)]
        return backlog, qc_pairs, acceptance


def field_name_to_accessor(name):
    parts = name.split('_')
    return parts[0] + ''.join(p[:1].upper() + p[1:] for p in parts[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help='mounted 21H461 root file system (read-only use)')
    sub = parser.add_subparsers(dest='command', required=True)
    derive = sub.add_parser('derive')
    derive.add_argument('--write', action='store_true')
    check = sub.add_parser('check')
    check.add_argument('--schema', default=str(SCHEMA_PATH))
    check.add_argument('--backlog', action='store_true', help='print every unclassified selector')
    args = parser.parse_args()
    target = Target(args.root)
    if args.command == 'derive':
        text = list_text(target.quartzcore_list())
        if args.write:
            LIST_PATH.write_text(text)
        sys.stdout.write(text)
        return 0
    schema = tomllib.loads(Path(args.schema).read_text())
    checker = Checker(schema, target)
    checker.check_list()
    claimed = checker.check_ops()
    checker.check_structs()
    local = checker.check_locals(claimed)
    backlog, qc_pairs, acceptance = checker.coverage(claimed, local)
    print('QuartzCore list: %d selectors, %d (protocol, selector) pairs, %d in MTL*CommandEncoder/MTLDevice/MTLDeviceSPI'
          % (len({s for _, s in qc_pairs}), len(qc_pairs), len(acceptance)))
    covered = [(p, s) for p, s in acceptance if (p, s) in claimed or (p, s) in local]
    print('acceptance pairs classified: %d of %d (%d opcode, %d guest-local)' % (
        len(covered), len(acceptance), sum((p, s) in claimed for p, s in acceptance),
        sum((p, s) in local for p, s in acceptance)))
    print('ops: %d methods mapped; guest-local: %d methods' % (len(claimed), len(local)))
    print('QuartzCore pairs outside that set left unclassified: %d' % len(checker.quartzcore_backlog))
    for protocol, selector in checker.quartzcore_backlog:
        print('  %s %s' % (protocol, selector))
    total = sum(len(v) for v in backlog.values())
    print('driver-protocol backlog (neither opcode nor guest-local): %d methods' % total)
    for protocol, selectors in sorted(backlog.items()):
        print('  %-44s %4d' % (protocol, len(selectors)))
        if args.backlog:
            for selector in selectors:
                print('      ' + selector)
    for message in checker.errors:
        print('error: ' + message)
    return 1 if checker.errors else 0


if __name__ == '__main__':
    sys.exit(main())
