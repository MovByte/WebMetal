"""Assemble the A10X device profile from method A and the cross-checks B1 and B2.

`collect` runs everything and returns a `Profile`; `render` turns it into the TOML text whose
values are numbers and booleans only and whose every value line carries its provenance as a
comment: the symbol and address that produced it, the method, and the second method's
verdict (`agrees`, or `single-method` with the reason). A disagreement between the methods
stops the build; it is never written out.
"""

import re
import struct
from pathlib import Path

from . import host_probe, sampler, static_decode
from .target import (ADD_DEVICE_SEQUENCE, NOT_EVALUATED, PARAMETERIZED, PIXEL_FORMAT_RANGE,
                     Extraction, Outcome, Target, argument_types, domain, return_type)

DRIVER_CLASSES = ('AGXA10XDevice', 'AGXA10FamilyDevice', 'IOGPUMetalDevice')
VALUE_RETURNS = {'B': 'B', 'Q': 'Q', 'q': 'q', 'I': 'I', 'f': 'f'}
SAMPLE_COUNTS = range(0, 17)
HEAP_LENGTHS = (0, 1, 2, 3, 4, 7, 8, 15, 16, 255, 256, 257, 4095, 4096, 4097, 16383, 16384, 16385,
                1 << 20, (1 << 32) + 1, (1 << 63) - 1)
# MTLResourceOptions: CPU cache mode (bits 0-3), storage mode (bits 4-7), hazard tracking (bits 8-9).
HEAP_OPTIONS = tuple(cache | storage << 4 | hazard << 8
                     for cache in (0, 1) for storage in (0, 1, 2, 3) for hazard in (0, 1, 2))
# MTLPixelFormatInfo words compared and written (offset, size); +0 is the name pointer.
INFO_WORDS = ((0x08, 4), (0x0c, 4), (0x10, 8), (0x18, 8), (0x20, 1), (0x21, 1), (0x22, 2),
              (0x24, 4), (0x28, 8), (0x30, 8))
# Flag bits of MTLPixelFormatInfo+0x08 whose meaning the target binary itself establishes.
NAMED_FLAG_BITS = {
    0: ('supported', 'tested before "not supported on this device" in '
                     '-[_MTLDevice minimumLinearTextureAlignmentForPixelFormat:]'),
    10: ('compressed', 'tested before the compressed-format refusal in '
                       '-[_MTLDevice minimumLinearTextureAlignmentForPixelFormat:]'),
    16: ('compatible_float', 'MTLPixelFormatCompatibilityString'),
    17: ('compatible_half', 'MTLPixelFormatCompatibilityString'),
    18: ('compatible_int', 'MTLPixelFormatCompatibilityString'),
    19: ('compatible_uint', 'MTLPixelFormatCompatibilityString'),
    20: ('compatible_short', 'MTLPixelFormatCompatibilityString'),
    21: ('compatible_ushort', 'MTLPixelFormatCompatibilityString'),
    29: ('compatible_ulong', 'MTLPixelFormatCompatibilityString'),
}


class BuildError(Exception):
    pass


def merge(first, second, inputs):
    """One outcome from the two sentinel runs."""
    if first.kind == second.kind == 'value':
        if first.value == second.value:
            return first
        return Outcome('runtime', None, 'derived from kernel input ' + ', '.join(sorted(inputs)))
    if first.kind != second.kind:
        raise BuildError(f'sentinel runs disagree in kind: {first} / {second}')
    return first


class Profile:
    def __init__(self):
        self.a = {}
        self.b = {}
        self.notes = {}


def ret_kind(types):
    rtype = return_type(types)
    if rtype in VALUE_RETURNS:
        return VALUE_RETURNS[rtype]
    if rtype == '{?=QQQ}':
        return 'size'
    if rtype.startswith('{') and re.search(r'b\d', rtype):
        return 'I'
    if rtype == '{?=II}' or rtype == '{?=QQ}':
        return 'pair'
    return None


def collect(cache_path, build_dir, llvm_bin, log=print):
    target = Target(cache_path)
    profile = Profile()
    p = profile.a
    log('method A: verifying the initial device state')
    p['initial_state'] = target.verify_initial_state()
    p['categories'] = target.verify_dispatch()
    log('method A: booting two sentinel runs')
    runs = [Extraction(target, 0), Extraction(target, 1)]
    inputs = runs[0].inputs
    p['inputs'] = inputs
    methods = target.protocol_methods()
    p['protocol_method_count'] = len(methods)

    log('method A: zero-argument getters')
    getters, skipped = {}, {}
    for selector, (proto, kind, types) in sorted(methods.items()):
        if selector in NOT_EVALUATED:
            skipped[selector] = NOT_EVALUATED[selector]
            continue
        if selector in PARAMETERIZED or argument_types(types):
            continue
        if selector == 'limits':
            skipped[selector] = 'returns a pointer to _limits; see [limits]'
            continue
        if ret_kind(types) is None:
            continue
        getters[selector] = (merge(runs[0].ask(selector, types), runs[1].ask(selector, types), inputs), types, proto)
    p['getters'] = getters
    log('method A: driver getters outside the protocols')
    driver_getters = {}
    for name in DRIVER_CLASSES:
        cls = next(c for c in target.chain if c.name == name)
        for method in cls.own_methods():
            selector = method.selector
            if selector in methods or selector in driver_getters or argument_types(method.types):
                continue
            if target.method(selector)[0].name != name or ret_kind(method.types) is None:
                continue
            driver_getters[selector] = (merge(runs[0].ask(selector, method.types),
                                              runs[1].ask(selector, method.types), inputs),
                                        method.types, name)
    p['driver_getters'] = driver_getters
    p['skipped'] = skipped

    log('method A: parameterized queries')
    parameterized = {}
    for selector in PARAMETERIZED:
        proto, kind, types = methods[selector]
        table = {}
        for arg in domain(selector):
            table[arg] = merge(runs[0].ask(selector, types, (arg,)), runs[1].ask(selector, types, (arg,)), inputs)
        parameterized[selector] = (table, types)
    p['parameterized'] = parameterized

    log('method A: limits')
    first, second = runs[0].limits(), runs[1].limits()
    limits = []
    for (name, kind, offset, v0), (_, _, _, v1) in zip(first, second):
        limits.append((name, kind, offset, Outcome('value', v0) if v0 == v1 else
                       Outcome('runtime', None, 'derived from kernel input ' + ', '.join(sorted(inputs)))))
    p['limits'] = limits

    log('method A: pixel formats')
    formats = {}
    for fmt in PIXEL_FORMAT_RANGE:
        o0, o1 = runs[0].pixel_format(fmt), runs[1].pixel_format(fmt)
        if o0.kind != 'value' or o1.kind != 'value' or o0.value != o1.value:
            raise BuildError(f'pixel format {fmt}: {o0} / {o1}')
        formats[fmt] = o0
    p['pixel_formats'] = formats
    p['pixel_format_function'] = target.cache.symbol('Metal', '_MTLPixelFormatGetInfoForDevice')

    log('method A: sample positions and heap sizes')
    p['sample_positions'] = {n: merge(runs[0].sample_positions(n), runs[1].sample_positions(n), inputs)
                             for n in SAMPLE_COUNTS}
    heap = {}
    for length in HEAP_LENGTHS:
        for options in HEAP_OPTIONS:
            heap[(length, options)] = merge(runs[0].heap_buffer(length, options),
                                            runs[1].heap_buffer(length, options), inputs)
    p['heap_buffer'] = heap
    # MTLPixelFormatBGRA8Unorm, 256 x 256, MTLTextureType2D, private storage.
    texture = (('setPixelFormat:', 80), ('setWidth:', 256), ('setHeight:', 256), ('setStorageMode:', 2))
    p['heap_texture'] = (merge(runs[0].heap_texture(texture), runs[1].heap_texture(texture), inputs), texture)
    p['sdk_queries'] = sorted({(pl, v, a) for run in runs for pl, v, a, _ in run.m.sdk_queries})
    p['environment'] = sorted({name for run in runs for name in run.m.environment_queries})
    p['where'] = {sel: target.where(sel) for sel in methods}
    p['definer'] = {sel: (target.method(sel)[0].name if target.method(sel)[0] else None) for sel in methods}

    log('method A: constant-sampler encoding')
    p['sampler'] = sampler.derive(target)
    log('method B3: constant samplers compiled by the host Metal compiler')
    profile.b['sampler'] = sampler.check_with_compiler(p['sampler'], llvm_bin, Path(build_dir) / 'sampler')
    log('method B2: static decoding of driver getters')
    profile.b['static'] = static_check(target, {**getters, **p['driver_getters']}, llvm_bin)
    profile.b['static_parameterized'] = static_parameterized(target, parameterized, llvm_bin)
    log('method B1: host Metal.framework')
    profile.b['host'] = host_check(target, profile, build_dir)
    profile.target = target
    return profile


def static_check(target, getters, llvm_bin):
    tool = static_decode.llvm_mc(llvm_bin)
    out = {}

    def resolve(selector):
        cls, method = target.method(selector)
        if method is None:
            raise static_decode.DecodeError(f'no implementation of {selector}')
        return method.imp

    for selector, (outcome, types, _) in list(getters.items()):
        cls, method = target.method(selector)
        if method is None or cls.name not in DRIVER_CLASSES:
            continue
        try:
            decoded = static_decode.decode_getter(target.cache, tool, method.imp, resolve)
        except static_decode.DecodeError as error:
            out[selector] = ('unreachable', str(error))
            continue
        out[selector] = ('value', decoded)
    selector = 'heapBufferSizeAndAlignWithLength:options:'
    cls, method = target.method(selector)
    try:
        out[selector] = ('value', static_decode.decode_getter(target.cache, tool, method.imp))
    except static_decode.DecodeError as error:
        out[selector] = ('unreachable', str(error))
    return out


def static_parameterized(target, parameterized, llvm_bin):
    """{selector: {arg: ('value', x0) | ('unreachable', reason)}} for driver-implemented queries."""
    tool = static_decode.llvm_mc(llvm_bin)
    out = {}

    def resolve(selector):
        cls, method = target.method(selector)
        if method is None:
            raise static_decode.DecodeError(f'no implementation of {selector}')
        return method.imp
    for selector, (table, types) in parameterized.items():
        cls, method = target.method(selector)
        if cls.name not in DRIVER_CLASSES:
            continue
        answers = {}
        for arg, outcome in table.items():
            try:
                decoded = static_decode.decode_getter(target.cache, tool, method.imp, resolve, arguments=(arg,))
                answers[arg] = ('value', decoded)
            except static_decode.DecodeError as error:
                answers[arg] = ('unreachable', str(error))
        out[selector] = answers
    return out


def static_value(decoded, kind):
    """The getter's value from a B2 decode, in the representation method A uses."""
    if kind == 'size':
        stores = decoded.get('x8_stores', {})
        if sorted(stores) != [0, 8, 16] or any(size != 8 for size, _ in stores.values()):
            return None
        return [stores[0][1], stores[8][1], stores[16][1]]
    if 'x0' not in decoded or isinstance(decoded['x0'], tuple):
        return None
    raw = decoded['x0']
    if kind == 'B':
        return bool(raw & 0xff) if raw & 0xff in (0, 1) else None
    if kind == 'I':
        return raw & 0xffffffff
    if kind == 'q':
        return raw - (1 << 64) if raw >> 63 else raw
    if kind == 'f':
        return struct.unpack('<f', struct.pack('<I', decoded['s0']))[0] if 's0' in decoded else None
    return raw


def host_check(target, profile, build_dir):
    p = profile.a
    getters, parameterized = p['getters'], p['parameterized']
    driver = {}
    seen = set()
    for name in DRIVER_CLASSES:
        cls = next(c for c in target.chain if c.name == name)
        for method in cls.own_methods():
            selector = method.selector
            if selector in seen or selector.startswith('.') or selector == 'dealloc' or selector.startswith('init'):
                continue
            seen.add(selector)
            definer, _ = target.method(selector)
            if definer.name != name:
                continue
            entry = {'types': method.types, 'kind': 'trap'}
            known = getters.get(selector) or p['driver_getters'].get(selector)
            if known and known[0].kind == 'value':
                kind = ret_kind(method.types)
                if kind in ('B', 'Q', 'q', 'I', 'f', 'size'):
                    entry.update(kind='value', ret=kind, value=known[0].value)
            elif selector in p['inputs']:
                entry.update(kind='value', ret=ret_kind(method.types), value=p['inputs'][selector]['value'])
            elif selector in parameterized:
                table, types = parameterized[selector]
                values = {str(arg): o.value for arg, o in table.items() if o.kind == 'value'}
                kind = ret_kind(types)
                if kind in ('B', 'Q'):
                    entry.update(kind='table', ret=kind, table=values)
            driver[selector] = entry
    framework_getters = {sel: ret_kind(types) for sel, (o, types, _) in getters.items()
                         if p['definer'][sel] == '_MTLDevice' and ret_kind(types)}
    framework_parameterized = {}
    for selector, (table, types) in parameterized.items():
        if p['definer'][selector] != '_MTLDevice':
            continue
        framework_parameterized[selector] = {
            'ret': ret_kind(types), 'args': [arg for arg, o in table.items() if o.kind == 'value']}
    request = {
        'driver': driver,
        'sequence': list(ADD_DEVICE_SEQUENCE),
        'getters': framework_getters,
        'parameterized': framework_parameterized,
        'pixel_formats': list(PIXEL_FORMAT_RANGE),
        'sample_counts': list(SAMPLE_COUNTS),
    }
    binary = host_probe.build(build_dir)
    response = host_probe.run(binary, request, build_dir)
    response['os'] = host_probe.host_identity()
    return response
