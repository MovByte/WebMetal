"""The 64-bit constant-sampler encoding (`air.sampler_state`) and its decoding.

Method A: the target's Metal.framework turns a constant sampler's 64-bit word into an
MTLSamplerDescriptor in `setDescriptorFromBitmask` (called by
`_MTLCreateConstantSamplerDescriptorsFromBitmasks` for pipeline reflection). That function
runs in emulation on chosen words: one word per bit finds which descriptor property each bit
feeds, a sweep of each field's codes gives its value table, and random words check that the
fields compose independently.

Method B: MSL `constexpr sampler` declarations compiled by this Mac's Metal compiler for
air64-apple-ios17.0 produce the words the encoder writes; each is decoded with the tables
from A and compared with the property values the MSL source names. The MSL argument names
map to the MTLSamplerDescriptor enumerators of the same name (MSL specification, "Samplers").
"""

import random
import re
import struct
import subprocess
from pathlib import Path

from .dyld import require
from .emu import EmulationError, ReportedFailure
from . import sdk

DECODER = '__ZL24setDescriptorFromBitmaskP20MTLSamplerDescriptoryb'
CALLER = '__MTLCreateConstantSamplerDescriptorsFromBitmasks'
# MTLSamplerDescriptor properties and the type of their getters.
PROPERTIES = (('sAddressMode', 'Q'), ('tAddressMode', 'Q'), ('rAddressMode', 'Q'), ('magFilter', 'Q'),
              ('minFilter', 'Q'), ('mipFilter', 'Q'), ('normalizedCoordinates', 'B'),
              ('compareFunction', 'Q'), ('maxAnisotropy', 'Q'), ('lodMinClamp', 'f'),
              ('lodMaxClamp', 'f'), ('borderColor', 'Q'))
FLOAT_FIELDS = ('lodMinClamp', 'lodMaxClamp')
VALID = 1 << 63


def half(code):
    return struct.unpack('<e', struct.pack('<H', code))[0]


def identical(a, b):
    """Equality that tells -0.0 from 0.0 (bitwise for floats). Two NaNs are identical: the
    payload and sign of a NaN from FCVT depend on FPCR.DN, which is process state, not part
    of the encoding."""
    if isinstance(a, float) or isinstance(b, float):
        if a != a and b != b:
            return True
        return struct.pack('<f', a) == struct.pack('<f', b)
    return a == b


class Decoder:
    """setDescriptorFromBitmask of the target, run in one emulated process."""

    def __init__(self, target):
        self.target = target
        self.m = m = target.machine()
        self.function = target.cache.symbol('Metal', DECODER)
        metal = target.metal
        self.descriptor_class = target.runtime.class_named(metal, 'MTLSamplerDescriptor').address
        self.getters = {}
        self.calls = 0

    def decode(self, word):
        """{property: value} for word, or ('failure', line)."""
        m, objc = self.m, self.m.objc
        descriptor = objc.send(objc.send(self.descriptor_class, 'alloc'), 'init')
        self.calls += 1
        try:
            m.call(self.function, (descriptor, word, 0))
        except ReportedFailure as failure:
            return ('failure', failure.line)
        out = {}
        for name, kind in PROPERTIES:
            imp = objc.dispatch(descriptor, name)
            self.getters[name] = (objc.rt.lookup(objc.class_of(descriptor), name)[0].name, imp)
            m.call(imp, (descriptor, objc.selector(name)))
            x0, _, q = m._last
            if kind == 'B':
                out[name] = bool(x0 & 0xff)
            elif kind == 'f':
                out[name] = struct.unpack('<f', struct.pack('<I', q[0] & 0xffffffff))[0]
            else:
                out[name] = x0
        return out


def derive(target, log=print, seed=0x21d461):
    """Field layout and value tables of the encoding, from the target's decoder."""
    d = Decoder(target)
    base = d.decode(VALID)
    require(isinstance(base, dict), f'decoder rejects the all-zero valid word: {base}')
    bits, failures = {}, {}
    for bit in range(63):
        result = d.decode(VALID | 1 << bit)
        if isinstance(result, tuple):
            failures[bit] = result[1]
            continue
        changed = [name for name in base if not identical(result[name], base[name])]
        require(len(changed) <= 1, f'bit {bit} changes {changed}')
        if changed:
            bits.setdefault(changed[0], []).append(bit)
    missing = d.decode(0)
    require(isinstance(missing, tuple), 'decoder accepts a word without bit 63')
    fields = {}
    for name, positions in bits.items():
        offset, width = positions[0], len(positions)
        require(positions == list(range(offset, offset + width)), f'{name} bits are not contiguous: {positions}')
        fields[name] = {'offset': offset, 'width': width}
    # A bit that alone changes nothing can still belong to a field: its single-bit code may
    # select the same value as code 0. Extend each field over such adjacent bits when some
    # code that uses the bit decodes differently (another value, or a failure) from the same
    # code without it.
    attributed = {b for f in fields.values() for b in range(f['offset'], f['offset'] + f['width'])} | set(failures)
    grown = True
    while grown:
        grown = False
        for name, field in fields.items():
            if name in FLOAT_FIELDS:
                continue
            for bit in (field['offset'] + field['width'], field['offset'] - 1):
                if bit < 0 or bit > 62 or bit in attributed:
                    continue
                low = min(bit, field['offset'])
                width = field['width'] + 1
                rel = bit - low
                differs = False
                for code in range(1 << width):
                    if not code >> rel & 1:
                        continue
                    with_bit = d.decode(VALID | code << low)
                    without = d.decode(VALID | (code & ~(1 << rel)) << low)
                    if isinstance(with_bit, tuple) or isinstance(without, tuple):
                        differs = with_bit != without
                    else:
                        differs = not identical(with_bit[name], without[name])
                    if differs:
                        break
                if differs:
                    field['offset'], field['width'] = low, width
                    attributed.add(bit)
                    grown = True
    for name, field in fields.items():
        offset, width = field['offset'], field['width']
        if name in FLOAT_FIELDS:
            codes = sorted(set([0, 1, 0x3c00, 0x3c01, 0x3bff, 0x7bff, 0x7c00, 0x7e00, 0x8000, 0x8001, 0xbc00,
                                0xfbff, 0xfc00, 0xffff, 0x0400, 0x03ff]) |
                           set(random.Random(seed + offset).sample(range(1 << width), 1024)))
            for code in codes:
                result = d.decode(VALID | code << offset)
                require(isinstance(result, dict), f'{name} code {code:#x} fails')
                expected = half(code)
                require(identical(result[name], expected), f'{name} code {code:#x}: {result[name]} is not binary16 {expected}')
            field.update(encoding='binary16', codes_checked=len(codes))
            continue
        values, failed = [], {}
        for code in range(1 << width):
            result = d.decode(VALID | code << offset)
            if isinstance(result, tuple):
                failed[code] = result[1]
                values.append(None)
            else:
                values.append(result[name])
        field.update(values=values, failures=failed)
    covered = sum(((1 << f['width']) - 1) << f['offset'] for f in fields.values())
    reserved = sum(1 << bit for bit in failures)
    unused = ((1 << 63) - 1) & ~covered & ~reserved
    # Random words with every field at a code it accepts decode to the composition of the
    # per-field tables (fields are independent).
    rng = random.Random(seed)
    composed = 0
    for _ in range(256):
        word, expected = VALID, {}
        for name, field in fields.items():
            if 'values' in field:
                choices = [c for c, v in enumerate(field['values']) if v is not None]
                code = rng.choice(choices)
                expected[name] = field['values'][code]
            else:
                code = rng.randrange(1 << field['width'])
                expected[name] = half(code)
            word |= code << field['offset']
        word |= rng.getrandbits(63) & unused
        result = d.decode(word)
        require(isinstance(result, dict), f'composed word {word:#x} fails: {result}')
        for name, value in expected.items():
            require(identical(result[name], value), f'word {word:#x}: {name} {result[name]} != {value}')
        composed += 1
    return {
        'fields': fields, 'reserved_bits': reserved, 'reserved_failures': failures,
        'missing_valid_failure': missing[1], 'unused_bits': unused, 'composed_checked': composed,
        'decoder': d.function, 'caller': target.cache.symbol('Metal', CALLER), 'calls': d.calls,
        'getters': d.getters,
    }


def decode_with(layout, word):
    """Decode word with the derived tables (pure Python)."""
    out = {}
    for name, field in layout['fields'].items():
        code = word >> field['offset'] & ((1 << field['width']) - 1)
        out[name] = half(code) if field.get('encoding') == 'binary16' else field['values'][code]
    return out


# -- method B: the host compiler as the encoder --------------------------------------------------

def _enum(name):
    return dict(sdk.enum_constants(name))


def msl_cases():
    """[(MSL constexpr sampler arguments, {property: expected value})]."""
    address = _enum('MTLSamplerAddressMode')
    minmag = _enum('MTLSamplerMinMagFilter')
    mip = _enum('MTLSamplerMipFilter')
    compare = _enum('MTLCompareFunction')
    border = _enum('MTLSamplerBorderColor')
    cases = []
    # address::clamp_to_border is not compared: for air64-apple-ios17.0 the compiler writes the
    # clamp_to_zero code for it (see `border_lowering`).
    for keyword, api in (('clamp_to_edge', 'ClampToEdge'), ('mirrored_repeat', 'MirrorRepeat'),
                         ('repeat', 'Repeat'), ('clamp_to_zero', 'ClampToZero')):
        value = address['MTLSamplerAddressMode' + api]
        for axis in ('s', 't', 'r'):
            cases.append((f'{axis}_address::{keyword}', {f'{axis}AddressMode': value}))
    for keyword, api in (('nearest', 'Nearest'), ('linear', 'Linear')):
        value = minmag['MTLSamplerMinMagFilter' + api]
        cases.append((f'mag_filter::{keyword}', {'magFilter': value}))
        cases.append((f'min_filter::{keyword}', {'minFilter': value}))
    for keyword, api in (('none', 'NotMipmapped'), ('nearest', 'Nearest'), ('linear', 'Linear')):
        cases.append((f'mip_filter::{keyword}', {'mipFilter': mip['MTLSamplerMipFilter' + api]}))
    for keyword, api in (('never', 'Never'), ('less', 'Less'), ('less_equal', 'LessEqual'),
                         ('greater', 'Greater'), ('greater_equal', 'GreaterEqual'), ('equal', 'Equal'),
                         ('not_equal', 'NotEqual'), ('always', 'Always')):
        cases.append((f'compare_func::{keyword}', {'compareFunction': compare['MTLCompareFunction' + api]}))
    for keyword, api in (('transparent_black', 'TransparentBlack'), ('opaque_black', 'OpaqueBlack'),
                         ('opaque_white', 'OpaqueWhite')):
        cases.append((f'border_color::{keyword}', {'borderColor': border['MTLSamplerBorderColor' + api]}))
    cases.append(('coord::normalized', {'normalizedCoordinates': True}))
    cases.append(('coord::pixel', {'normalizedCoordinates': False}))
    for n in range(1, 17):
        cases.append((f'max_anisotropy({n})', {'maxAnisotropy': n}))
    for low, high in ((0.0, 1000.0), (1.0, 7.5), (0.25, 0.5), (2.0, 15.0), (0.0, 0.0)):
        cases.append((f'lod_clamp({low!r}, {high!r})', {'lodMinClamp': low, 'lodMaxClamp': high}))
    return cases


def compile_words(cases, llvm_bin, work):
    """The 64-bit word the host compiler emits for each case."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    lines = ['#include <metal_stdlib>', 'using namespace metal;']
    for i, (args, _) in enumerate(cases):
        lines.append(f'constexpr sampler s{i}({args});')
    # A function may use at most 16 samplers; one kernel per group of 8 keeps the order of the
    # sample calls in the module equal to the order of the cases.
    for group in range(0, len(cases), 8):
        lines.append(f'kernel void k{group}(texture2d<float> t [[texture(0)]], device float4 *o [[buffer(0)]]) {{')
        for i in range(group, min(group + 8, len(cases))):
            lines.append(f'  o[{i}] = t.sample(s{i}, float2(0.5), level(0));')
        lines.append('}')
    source = work / 'constant_samplers.metal'
    source.write_text('\n'.join(lines) + '\n')
    air = work / 'constant_samplers.air'
    run = subprocess.run(['xcrun', '-sdk', 'iphoneos', 'metal', '-target', 'air64-apple-ios17.0', '-c',
                          str(source), '-o', str(air)], capture_output=True, text=True, timeout=600)
    if run.returncode:
        raise EmulationError(f'metal failed: {run.stderr[-3000:]}')
    text = subprocess.run([str(Path(llvm_bin) / 'llvm-dis'), str(air), '-o', '-'], capture_output=True,
                          text=True, timeout=600, check=True).stdout
    constants = dict(re.findall(r'^@(__air_sampler_state[.\d]*) = internal addrspace\(2\) constant i64 (-?\d+)',
                                text, re.M))
    # Each sample call names the sampler global it uses; map the calls back to the cases in order.
    used = re.findall(r'@air\.sample_texture_2d[^(]*\([^@]*@(__air_sampler_state[.\d]*)', text)
    require(len(used) == len(cases), f'{len(used)} sample calls for {len(cases)} samplers')
    return [int(constants[name]) & (1 << 64) - 1 for name in used], compiler_version()


def compiler_version():
    out = subprocess.run(['xcrun', '-sdk', 'iphoneos', 'metal', '--version'], capture_output=True, text=True)
    return out.stdout.splitlines()[0].strip() if out.stdout else 'unknown'


def check_with_compiler(layout, llvm_bin, work):
    """Confirmed (property, code) pairs and lod values; raises on any disagreement."""
    cases = msl_cases()
    border = [('s_address::clamp_to_border', None), ('s_address::clamp_to_zero', None)]
    words, version = compile_words(cases + border, llvm_bin, work)
    confirmed, lods = set(), set()
    reserved = layout['reserved_bits']
    for (args, expected), word in zip(cases, words):
        require(word & VALID and not word & reserved, f'{args}: compiled word {word:#x} has invalid fixed bits')
        decoded = decode_with(layout, word)
        for name, value in expected.items():
            require(identical(decoded[name], value), f'{args}: word {word:#x} decodes {name}={decoded[name]}, MSL says {value}')
            field = layout['fields'][name]
            code = word >> field['offset'] & ((1 << field['width']) - 1)
            if field.get('encoding') == 'binary16':
                lods.add((name, code))
            else:
                confirmed.add((name, code))
    field = layout['fields']['sAddressMode']
    border_code, zero_code = (w >> field['offset'] & ((1 << field['width']) - 1) for w in words[len(cases):])
    return {'confirmed': confirmed, 'lods': lods, 'version': version, 'cases': len(cases),
            'border_lowering': (border_code, zero_code)}
