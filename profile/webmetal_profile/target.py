"""Method A: the A10X device profile computed by executing the target's own code.

The Metal driver classes of the 21H461 shared cache run under emulation (`emu.Machine`) on a
fresh `AGXA10XDevice` instance. The instance goes through the sequence `MTLAddDevice`
performs after `initWithAcceleratorPort:` (`initLimits`, `initFeatureQueries`,
`initWorkarounds`), then every capability query of the `MTLDevice`/`MTLDeviceSPI`
protocols is sent to it and its answer recorded.

The instance's storage is tracked byte by byte. Device state that only the kernel supplies
(the accelerator connection, the IOGPU device, sizes read from the registry) is never
invented: a query that reads it is recorded as `runtime` with the instance variable it read.
Driver methods that return such state, and that the framework calls while computing other
answers, are answered with sentinel values in two runs with different sentinels; any answer
that differs between the runs is recorded as derived from that kernel input.
"""

import bisect
import re
import struct

from .dyld import Cache, CacheError, require
from .emu import EmulationError, Machine, ReportedFailure, UninitializedRead
from .objc import Runtime
from . import sdk

CACHE_UUID = '590d839a82d230f1a17d0cc053420f73'
EXECUTED_IMAGES = ('Metal', 'AGXMetalA10', 'IOGPU')
DEVICE_IMAGE, DEVICE_CLASS = 'AGXMetalA10', 'AGXA10XDevice'
ADD_DEVICE_SEQUENCE = ('initLimits', 'initFeatureQueries', 'initWorkarounds')
# Instance variables that -[_MTLDevice init] leaves at the zero storage class_createInstance
# gives them. Checked by `verify_initial_state`.
NULL_AFTER_INIT = ('_supportedGPUFamilies', '_featureQueries')
SENTINELS = ((0x5e5e0000 + 0x1000, 0x10000), (0x7b7b000000 + 0x2000, 0x3000000))

SCALAR = {'B': (1, False), 'c': (1, True), 'C': (1, False), 's': (2, True), 'S': (2, False),
          'i': (4, True), 'I': (4, False), 'l': (8, True), 'L': (8, False), 'q': (8, True),
          'Q': (8, False)}


def return_type(types):
    """The return type encoding of an ObjC method type string (without frame offsets)."""
    t = types.lstrip('rnNoORV')
    if t[0] == '{':
        depth = 0
        for i, ch in enumerate(t):
            depth += ch == '{'
            depth -= ch == '}'
            if depth == 0:
                return t[:i + 1]
    if t[0] == '^':
        return '^' + return_type(t[1:])
    return t[0]


def argument_types(types):
    """Argument encodings after self and _cmd, without frame offsets."""
    body = re.sub(r'\d+', '', types)
    rtype = return_type(types)
    rest = body[len(re.sub(r'\d+', '', rtype)) + (len(body) - len(body.lstrip('rnNoORV'))):]
    rest = rest[2:] if rest.startswith('@:') else rest
    out, i = [], 0
    while i < len(rest):
        ch = rest[i]
        if ch in 'rnNoORV':
            i += 1
            continue
        if ch == '{':
            depth, j = 0, i
            while True:
                depth += rest[j] == '{'
                depth -= rest[j] == '}'
                j += 1
                if depth == 0:
                    break
            out.append(rest[i:j])
            i = j
        elif ch == '^':
            j = i + 1
            if rest[j] == '{':
                depth = 0
                while True:
                    depth += rest[j] == '{'
                    depth -= rest[j] == '}'
                    j += 1
                    if depth == 0:
                        break
            else:
                j += 1
            out.append(rest[i:j])
            i = j
        elif ch == '@' and rest[i + 1:i + 2] == '?':
            out.append('@?')
            i += 2
        else:
            out.append(ch)
            i += 1
    return out


def struct_fields(encoding):
    """[(name, type)] of a struct encoding with quoted field names."""
    inner = encoding[encoding.index('=') + 1:-1]
    return re.findall(r'"([^"]+)"([A-Za-z])', inner)


class Outcome:
    __slots__ = ('kind', 'value', 'detail')

    def __init__(self, kind, value=None, detail=''):
        self.kind, self.value, self.detail = kind, value, detail

    def __repr__(self):
        return f'Outcome({self.kind}, {self.value!r}, {self.detail!r})'

    def __eq__(self, other):
        return isinstance(other, Outcome) and (self.kind, self.value) == (other.kind, other.value)


class Target:
    def __init__(self, cache_path):
        self.cache = Cache(cache_path)
        require(self.cache.uuid.hex() == CACHE_UUID, f'cache UUID {self.cache.uuid.hex()} is not 21H461')
        self.runtime = Runtime(self.cache)
        self.metal = self.cache.image('Metal')
        self.images = [self.cache.image(name) for name in EXECUTED_IMAGES]
        self.device_class = self.runtime.class_named(self.cache.image(DEVICE_IMAGE), DEVICE_CLASS)
        self.chain = self.runtime.chain(self.device_class)

    # -- environment --------------------------------------------------------------------------
    def machine(self):
        m = Machine(self.cache, self.runtime, self.images)
        for name in ('_MTLReleaseAssertionFailure', '_MTLReportFailure'):
            m.intercept(self.cache.symbol('Metal', name))
        return m

    def method(self, selector):
        cls, method = self.runtime.lookup(self.device_class, selector)
        return cls, method

    def where(self, selector):
        cls, method = self.method(selector)
        return f'-[{cls.name} {selector}] {method.imp:#x}' if method else 'unimplemented'

    def protocol_methods(self):
        """{selector: (protocol, kind, types)} of MTLDevice and MTLDeviceSPI (NSObject excluded)."""
        out = {}
        for name in ('MTLDevice', 'MTLDeviceSPI'):
            for proto in self.runtime.protocol_closure(self.runtime.protocol_named(self.metal, name)):
                if proto.name == 'NSObject':
                    continue
                for kind, selector, types, full in proto.methods:
                    if kind.startswith('class'):
                        continue
                    out.setdefault(selector, (proto.name, kind, types))
        return out

    # -- boot ---------------------------------------------------------------------------------
    def boot(self, sentinel_set):
        """A machine and a tracked device after MTLAddDevice's post-init sequence.

        A driver method that reads unset device state while the sequence runs becomes a kernel
        input: the sequence restarts with that method answered by a sentinel value.
        """
        inputs = {}
        base, step = SENTINELS[sentinel_set]
        for _ in range(32):
            m = self.machine()
            device = m.new_object(self.device_class, track=True, label=f'{DEVICE_CLASS} instance')
            for name in NULL_AFTER_INIT:
                cls, ivar = self.runtime.ivar(self.device_class, name)
                device.written[ivar.offset:ivar.offset + ivar.size] = b'\1' * ivar.size
            for index, (selector, info) in enumerate(sorted(inputs.items())):
                value = base + step * index
                info['value'] = value
                m.objc.overrides[(device.address, selector)] = (lambda v: lambda _: v)(value)
            try:
                for selector in ADD_DEVICE_SEQUENCE:
                    m.objc.send(device.address, selector)
                return m, device, inputs
            except UninitializedRead as error:
                selector = self._method_at(error.pc)
                if selector is None or selector in inputs:
                    raise EmulationError(f'boot reads unset device state outside a driver getter: {error}')
                inputs[selector] = dict(read=error.where, pc=error.pc, at=m.describe(error.pc))
        raise EmulationError('boot did not converge')

    def _method_at(self, pc):
        """Selector of the device-chain method containing pc, if any."""
        image, name, _ = self.cache.symbolize(pc)
        match = re.fullmatch(r'-\[(\w+) (\S+)\]', name or '')
        if not match or match.group(1) not in {c.name for c in self.chain}:
            return None
        return match.group(2)

    def verify_initial_state(self):
        """Evidence that NULL_AFTER_INIT ivars are zero when MTLAddDevice runs.

        -[_MTLDevice init] is the framework's initializer on the device creation path; it must
        contain no store to the ivar's offset from self, and Metal must define no setter for it.
        The driver images (IOGPU, AGXMetalA10) run the rest of initWithAcceleratorPort:; they
        can address the ivar only through its offset variable, so neither may reference that
        variable (by GOT pointer or by an ADRP+ADD/LDR pair).
        """
        evidence = {}
        base = self.runtime.class_named(self.metal, '_MTLDevice')
        init = self.runtime.lookup(base, 'init')[1]
        stores = init_stores(self.cache, init.imp)
        for name in NULL_AFTER_INIT:
            _, ivar = self.runtime.ivar(base, name)
            symbol = f'_OBJC_IVAR_$__MTLDevice.{name}'
            variable = self.cache.symbol('Metal', symbol)
            setter = 'set' + name.lstrip('_')[0].upper() + name.lstrip('_')[1:] + ':'
            require(self.runtime.lookup(base, setter)[1] is None, f'_MTLDevice defines {setter}')
            hits = [s for s in stores if s[0] < ivar.offset + ivar.size and ivar.offset < s[0] + s[1]]
            require(not hits, f'-[_MTLDevice init] stores to {name}')
            for image_name in ('IOGPU', 'AGXMetalA10'):
                refs = references(self.cache, self.cache.image(image_name), variable)
                require(not refs, f'{image_name} references {symbol} at {[hex(r) for r in refs]}')
            evidence[name] = (f'-[_MTLDevice init] {init.imp:#x}: {len(stores)} stores to self, none at '
                              f'+{ivar.offset:#x}; no {setter}; {symbol} ({variable:#x}) is not '
                              f'referenced by IOGPU or AGXMetalA10')
        return evidence

    def verify_dispatch(self):
        """Number of categories, in every image of the cache, on a class of the device chain.

        Message dispatch in method A uses the classes' own method lists; a category from
        another image would change it when that image is loaded.
        """
        classes = {c.address for c in self.chain if c.name != 'NSObject'}
        found = self.runtime.categories_on(classes)
        require(not found, f'categories on the device chain: {[(p, n) for p, n, _, _ in found]}')
        return len(found), len(self.cache.images)

    # -- queries ------------------------------------------------------------------------------
    def snapshot(self, m, device):
        return m.read(device.address, device.size), bytes(device.written)

    def restore(self, m, device, snap):
        data, written = snap
        m.uc.mem_write(device.address, data)
        device.written[:] = written

    def ask(self, m, device, selector, types, args=()):
        """Send one query; decode its return value from the method's type encoding."""
        cls, method = self.method(selector)
        if method is None:
            return Outcome('unimplemented', None, 'no implementation in the class chain')
        rtype = return_type(types)
        x8 = None
        if rtype.startswith('{') and struct_size(rtype) > 16:
            x8 = m.malloc(struct_size(rtype))
        try:
            imp = m.objc.dispatch(device.address, selector)
            m.call(imp, (device.address, m.objc.selector(selector)) + tuple(args), x8=x8)
        except UninitializedRead as error:
            return Outcome('runtime', None, f'reads {error.where} at {m.describe(error.pc)}')
        except ReportedFailure as error:
            return Outcome('failure', error.line, str(error))
        except EmulationError as error:
            return Outcome('error', None, str(error))
        x0, x1, q = m._last
        return Outcome('value', decode_return(m, rtype, x0, x1, q, x8), f'-[{cls.name} {selector}] {method.imp:#x}')


def struct_size(encoding):
    """Size of a struct of scalars/bitfields as returned by value (arm64 natural layout)."""
    fields = re.findall(r'b(\d+)|([BcCsSiIlLqQfd])', encoding[encoding.index('=') + 1:-1])
    size = 0
    bits = 0
    for width, kind in fields:
        if width:
            bits += int(width)
            continue
        if bits:
            size += (bits + 7) // 8
            bits = 0
        n = {'B': 1, 'c': 1, 'C': 1, 's': 2, 'S': 2, 'i': 4, 'I': 4, 'f': 4, 'd': 8,
             'l': 8, 'L': 8, 'q': 8, 'Q': 8}[kind]
        size = (size + n - 1) // n * n + n
    if bits:
        size += (bits + 7) // 8
    return size


def decode_return(m, rtype, x0, x1, q, x8):
    if rtype in SCALAR:
        size, signed = SCALAR[rtype]
        value = x0 & ((1 << 8 * size) - 1)
        if rtype == 'B':
            require(value in (0, 1), f'BOOL return {value:#x}')
            return bool(value)
        if signed and value >> (8 * size - 1):
            value -= 1 << 8 * size
        return value
    if rtype == 'f':
        return struct.unpack('<f', struct.pack('<I', q[0] & 0xffffffff))[0]
    if rtype == 'd':
        return struct.unpack('<d', struct.pack('<Q', q[0] & (1 << 64) - 1))[0]
    if rtype.startswith('{'):
        size = struct_size(rtype)
        if size <= 16:
            raw = struct.pack('<QQ', x0, x1)[:size]
        else:
            raw = m.read(x8, size)
        if 'b' in rtype and re.search(r'b\d', rtype):
            return int.from_bytes(raw, 'little')
        kinds = re.findall(r'[BcCsSiIlLqQfd]', re.sub(r'"[^"]*"', '', rtype[rtype.index('=') + 1:-1]))
        fmt = '<' + ''.join({'B': 'B', 'c': 'b', 'C': 'B', 's': 'h', 'S': 'H', 'i': 'i', 'I': 'I',
                             'l': 'q', 'L': 'Q', 'q': 'q', 'Q': 'Q', 'f': 'f', 'd': 'd'}[k] for k in kinds)
        return list(struct.unpack(fmt, raw[:struct.calcsize(fmt)]))
    if rtype.startswith('^'):
        return x0
    raise EmulationError(f'return type {rtype} is not decoded')


def init_stores(cache, address):
    """(offset, size) of every store through x19 after `mov x19, x0` in an -init method.

    -[_MTLDevice init] calls [super init] through objc_msgSendSuper2, keeps the result in x19
    and stores the instance variables relative to it. The whole function (to the next symbol)
    is decoded. Load/store encodings decoded: STR/LDR (unsigned offset), STUR/LDUR, STP/LDP
    (signed offset), integer and SIMD forms. Any other load/store-class instruction based on
    x19 fails the check instead of being skipped.
    """
    image = cache.image_at(address)
    image.symbols
    addresses = image._by_address[0]
    end = addresses[bisect.bisect_right(addresses, address)]
    words = struct.unpack(f'<{(end - address) // 4}I', cache.read(address, end - address))
    self_seen, stores = False, []
    for i, w in enumerate(words):
        if w == 0xaa0003f3:              # mov x19, x0
            self_seen = True
            continue
        if not self_seen or (w >> 5) & 31 != 19 or w & 0x0a000000 != 0x08000000:
            continue
        vector = bool(w & (1 << 26))
        if w & 0x3b000000 == 0x39000000:            # LDR/STR (immediate, unsigned offset)
            opc = (w >> 22) & 3
            scale = (((opc >> 1) << 2) | (w >> 30)) if vector else w >> 30
            if not opc & 1 if vector else opc == 0:
                stores.append((((w >> 10) & 0xfff) << scale, 1 << scale))
        elif w & 0x3b200c00 == 0x38000000:          # LDUR/STUR
            opc = (w >> 22) & 3
            scale = (((opc >> 1) << 2) | (w >> 30)) if vector else w >> 30
            imm = (w >> 12) & 0x1ff
            if not opc & 1 if vector else opc == 0:
                stores.append((imm - 0x200 if imm & 0x100 else imm, 1 << scale))
        elif w & 0x3b800000 == 0x29000000:          # LDP/STP (signed offset)
            opc = w >> 30
            scale = 2 + opc if vector else 2 + (opc >> 1)
            imm = (w >> 15) & 0x7f
            imm = (imm - 0x80 if imm & 0x40 else imm) << scale
            if not (w >> 22) & 1:
                stores.append((imm, 2 << scale))
        else:
            raise CacheError(f'unclassified load/store through x19 at {address + 4 * i:#x}: {w:#010x}')
    require(self_seen, f'no `mov x19, x0` in -init at {address:#x}')
    return stores


def references(cache, image, target):
    """Addresses in image that refer to target: data pointers equal to it, or ADRP pairs."""
    found = []
    image._parse()
    for (segment, section), (start, size) in image.sections.items():
        if segment.startswith('__DATA') or segment.startswith('__AUTH'):
            if section in ('__got', '__auth_got', '__const', '__objc_ivar', '__data'):
                for at in range(start, start + size - 7, 8):
                    m = cache.mapping(at)
                    if m is None or m.slide is None:
                        continue
                    page, offset = divmod(at - m.address, m.slide.page_size)
                    if m.rebases(page).get(offset) == target:
                        found.append(at)
    for _, start, size in image.section('__TEXT', '__text'):
        words = struct.unpack(f'<{size // 4}I', cache.read(start, size))
        pages = {}
        for i, w in enumerate(words):
            pc = start + 4 * i
            if w & 0x9f000000 == 0x90000000:                      # ADRP
                imm = ((w >> 5) & 0x7ffff) << 2 | (w >> 29) & 3
                imm -= (1 << 21) if imm & (1 << 20) else 0
                pages[w & 31] = (pc & ~0xfff) + (imm << 12)
                continue
            rn = (w >> 5) & 31
            if rn not in pages:
                continue
            if w & 0xff800000 == 0x91000000:                      # ADD (immediate), 64-bit
                address = pages[rn] + ((w >> 10) & 0xfff) * (4096 if (w >> 22) & 1 else 1)
            elif w & 0x3f000000 == 0x39000000:                    # LDR*/STR* (unsigned offset), integer
                address = pages[rn] + ((w >> 10) & 0xfff) * (1 << (w >> 30))
            else:
                pages.pop(w & 31, None)
                continue
            if address == target:
                found.append(pc)
    return found


# -- extraction -------------------------------------------------------------------------------

# Integer-argument queries and their argument domains. A domain names the SDK enum whose
# enumerators are evaluated, or an explicit range; `extra` adds the values just past the end.
PARAMETERIZED = {
    'supportsFamily:': ('MTLGPUFamily', None),
    'supportsFeatureSet:': ('MTLFeatureSet', None),
    'deviceSupportsFeatureSet:': ('MTLFeatureSet', None),
    'deviceOrFeatureProfileSupportsFeatureSet:': ('MTLFeatureSet', None),
    'supportsTextureSampleCount:': (None, range(0, 65)),
    'supportsSampleCount:': (None, range(0, 65)),
    'supportsPrimitiveType:': ('MTLPrimitiveType', range(0, 9)),
    'supportsVertexAmplificationCount:': (None, range(0, 17)),
    'supportsRasterizationRateMapWithLayerCount:': (None, range(0, 17)),
    'supportsCounterSampling:': ('MTLCounterSamplingPoint', range(0, 6)),
    'supportsTextureWriteRoundingMode:': (None, range(0, 4)),
    'sparseTileSizeInBytesForSparsePageSize:': ('MTLSparsePageSize', range(100, 105)),
    'minimumLinearTextureAlignmentForPixelFormat:': ('MTLPixelFormat', None),
    'minLinearTextureAlignmentForPixelFormat:': ('MTLPixelFormat', None),
    'minimumTextureBufferAlignmentForPixelFormat:': ('MTLPixelFormat', None),
}
# Protocol queries that are not evaluated, with the reason.
NOT_EVALUATED = {
    'resourcePatchingTypeForResourceType:': 'indexes a table with the unchecked argument; no header defines its domain',
    'isVendorSliceCompatibleWithDeploymentTarget:platform:sdkVersion:compilerPluginVersion:':
        'binary-archive slice compatibility for build metadata, not a device capability',
    'isCompatibleWithAccelerationStructure:': 'takes an acceleration-structure size descriptor',
    'setCompilerProcessesCount:': 'a setter',
    'reserveGPUAddressRange:': 'reserves address space (an action)',
}
PIXEL_FORMAT_RANGE = range(0, 0x300)
PIXEL_FORMAT_INFO_SIZE = 0x38


def domain(selector):
    enum, explicit = PARAMETERIZED[selector]
    values = set(explicit or ())
    if enum:
        values |= {value for _, value in sdk.enum_constants(enum)}
    return sorted(values)


class Extraction:
    """One run of method A with one sentinel set for the kernel inputs."""

    def __init__(self, target, sentinel_set):
        self.target = target
        self.m, self.device, self.inputs = target.boot(sentinel_set)
        self.snap = target.snapshot(self.m, self.device)

    def fresh(self):
        self.target.restore(self.m, self.device, self.snap)

    def ask(self, selector, types, args=()):
        self.fresh()
        return self.target.ask(self.m, self.device, selector, types, args)

    def limits(self):
        t = self.target
        cls, ivar = t.runtime.ivar(t.device_class, '_limits')
        fields = struct_fields(ivar.type)
        data = self.m.read(self.device.address + ivar.offset, ivar.size)
        out, at = [], 0
        for name, kind in fields:
            size = 8 if kind == 'Q' else 4
            at = (at + size - 1) // size * size
            raw = data[at:at + size]
            value = struct.unpack('<f', raw)[0] if kind == 'f' else int.from_bytes(raw, 'little')
            out.append((name, kind, ivar.offset + at, value))
            at += size
        require(at <= ivar.size, '_limits fields exceed the ivar size')
        return out

    def pixel_format(self, fmt):
        self.fresh()
        m = self.m
        out = m.malloc(PIXEL_FORMAT_INFO_SIZE)
        function = self.target.cache.symbol('Metal', '_MTLPixelFormatGetInfoForDevice')
        try:
            m.call(function, (self.device.address, fmt), x8=out)
        except UninitializedRead as error:
            return Outcome('runtime', None, f'reads {error.where}')
        except ReportedFailure as error:
            return Outcome('failure', error.line, str(error))
        except EmulationError as error:
            return Outcome('error', None, str(error))
        return Outcome('value', m.read(out, PIXEL_FORMAT_INFO_SIZE), f'_MTLPixelFormatGetInfoForDevice {function:#x}')

    def sample_positions(self, count):
        self.fresh()
        m = self.m
        buffer = m.malloc(8 * max(count, 1))
        selector = 'getDefaultSamplePositions:count:'
        cls, method = self.target.method(selector)
        try:
            m.call(method.imp, (self.device.address, m.objc.selector(selector), buffer, count))
        except ReportedFailure as error:
            return Outcome('failure', error.line, str(error))
        except EmulationError as error:
            return Outcome('error', None, str(error))
        values = struct.unpack(f'<{2 * count}f', m.read(buffer, 8 * count)) if count else ()
        return Outcome('value', list(values), f'-[{cls.name} {selector}] {method.imp:#x}')

    def heap_texture(self, settings):
        """heapTextureSizeAndAlignWithDescriptor: for a descriptor built with the given setters."""
        self.fresh()
        m, objc = self.m, self.m.objc
        selector = 'heapTextureSizeAndAlignWithDescriptor:'
        cls, method = self.target.method(selector)
        descriptor_class = self.target.runtime.class_named(self.target.metal, 'MTLTextureDescriptor').address
        try:
            descriptor = objc.send(objc.send(descriptor_class, 'alloc'), 'init')
            for setter, value in settings:
                objc.send(descriptor, setter, value)
            m.call(method.imp, (self.device.address, m.objc.selector(selector), descriptor))
        except UninitializedRead as error:
            return Outcome('runtime', None, f'reads {error.where} at {m.describe(error.pc)}')
        except ReportedFailure as error:
            return Outcome('failure', error.line, str(error))
        except EmulationError as error:
            return Outcome('error', None, str(error))
        x0, x1, _ = m._last
        return Outcome('value', (x0, x1), f'-[{cls.name} {selector}] {method.imp:#x}')

    def heap_buffer(self, length, options):
        self.fresh()
        selector = 'heapBufferSizeAndAlignWithLength:options:'
        cls, method = self.target.method(selector)
        try:
            self.m.call(method.imp, (self.device.address, self.m.objc.selector(selector), length, options))
        except EmulationError as error:
            return Outcome('error', None, str(error))
        x0, x1, _ = self.m._last
        return Outcome('value', (x0, x1), f'-[{cls.name} {selector}] {method.imp:#x}')
