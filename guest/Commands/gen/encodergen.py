#!/usr/bin/env python3
"""Generate the encoder command methods of the guest plugin from the wire schema.

Each Metal encoder command maps to one stream record that carries every argument verbatim (wire/README.md). For every
driver selector of an encoder role (guest/Plugin/roles.toml) that some stream op of the schema lists among its Metal
methods, under a protocol of the role's encoder kind, this script writes the method that fills the op's record from the
arguments and appends it to the encoder's stream. The method's parameter types follow the 21H461 type encoding of the
driver selector (guest/Generated/WebMetalTargetSurface.h), so the plugin's method has the driver's signature.

Methods whose behavior is more than writing one record (encoder life cycle, labels, debug groups) are written by hand in
guest/Commands/WebMetalEncoders.mm and are excluded here. An argument the script cannot convert (a descriptor structure,
a property-list value) makes it fail rather than guess.

    python3 -B -I guest/Commands/gen/encodergen.py           write guest/Commands/WebMetalEncoderRecords.mm
    python3 -B -I guest/Commands/gen/encodergen.py --check   fail if the committed file differs
"""
import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / 'wire/schema/webmetal.toml'
SURFACE = ROOT / 'guest/Generated/WebMetalTargetSurface.h'
OUTPUT = ROOT / 'guest/Commands/WebMetalEncoderRecords.mm'

# Encoder roles: plugin class, driver class, and the protocols whose ops it serves.
ROLES = [
    ('WebMetalRenderCommandEncoder', 'AGXA10FamilyRenderContext', ('MTLRenderCommandEncoder', 'MTLRenderCommandEncoderSPI')),
    ('WebMetalComputeCommandEncoder', 'AGXA10FamilyComputeContext', ('MTLComputeCommandEncoder', 'MTLComputeCommandEncoderSPI')),
    ('WebMetalBlitCommandEncoder', 'AGXA10FamilyBlitContext', ('MTLBlitCommandEncoder', 'MTLBlitCommandEncoderSPI')),
    ('WebMetalParallelRenderCommandEncoder', 'AGXA10FamilyThreadedRenderPass',
     ('MTLParallelRenderCommandEncoder', 'MTLParallelRenderCommandEncoderSPI')),
]

# Written by hand (WebMetalEncoders.mm).
HAND_WRITTEN = {'endEncoding', 'setLabel:', 'pushDebugGroup:', 'popDebugGroup', 'insertDebugSignpost:',
                'renderCommandEncoder', 'sampleCountersInBuffer:atSampleIndex:withBarrier:'}

# Object kinds: the protocol of the parameter type.
KIND_PROTOCOLS = {
    'Buffer': 'MTLBuffer', 'Texture': 'MTLTexture', 'Resource': 'MTLResource', 'Heap': 'MTLHeap',
    'SamplerState': 'MTLSamplerState', 'DepthStencilState': 'MTLDepthStencilState',
    'RenderPipelineState': 'MTLRenderPipelineState', 'ComputePipelineState': 'MTLComputePipelineState',
    'Fence': 'MTLFence', 'Event': 'MTLEvent', 'IndirectCommandBuffer': 'MTLIndirectCommandBuffer',
    'CounterSampleBuffer': 'MTLCounterSampleBuffer', 'AccelerationStructure': 'MTLAccelerationStructure',
    'IntersectionFunctionTable': 'MTLIntersectionFunctionTable', 'VisibleFunctionTable': 'MTLVisibleFunctionTable',
    'RasterizationRateMap': 'MTLRasterizationRateMap',
}

# Plain structures: Metal type, encoding, and the wire fields with the Metal members they take.
STRUCTS = {
    'Origin': ('MTLOrigin', '{?=QQQ}', [('x', 'x'), ('y', 'y'), ('z', 'z')]),
    'Size': ('MTLSize', '{?=QQQ}', [('width', 'width'), ('height', 'height'), ('depth', 'depth')]),
    'Range': ('NSRange', '{_NSRange=QQ}', [('location', 'location'), ('length', 'length')]),
    'Viewport': ('MTLViewport', '{?=dddddd}', [('originX', 'originX'), ('originY', 'originY'), ('width', 'width'),
                                               ('height', 'height'), ('znear', 'znear'), ('zfar', 'zfar')]),
    'ScissorRect': ('MTLScissorRect', '{?=QQQQ}', [('x', 'x'), ('y', 'y'), ('width', 'width'), ('height', 'height')]),
    'ClearColor': ('MTLClearColor', '{?=dddd}', [('red', 'red'), ('green', 'green'), ('blue', 'blue'),
                                                 ('alpha', 'alpha')]),
    'SamplePosition': ('MTLSamplePosition', '{?=ff}', [('x', 'x'), ('y', 'y')]),
    'VertexAmplificationViewMapping': ('MTLVertexAmplificationViewMapping', '{?=II}',
                                       [('viewportArrayIndexOffset', 'viewportArrayIndexOffset'),
                                        ('renderTargetArrayIndexOffset', 'renderTargetArrayIndexOffset')]),
}
REGION = ('MTLRegion', '{?={?=QQQ}{?=QQQ}}')

SCALARS = {
    'u8': ('C', 'uint8_t'), 'u16': ('S', 'uint16_t'), 'u32': ('I', 'uint32_t'), 'i32': ('i', 'int32_t'),
    'u64': ('Q', 'NSUInteger'), 'i64': ('q', 'NSInteger'), 'f32': ('f', 'float'), 'f64': ('d', 'double'),
    'bool': ('B', 'BOOL'),
}

# Metal enums the schema names on a field: the parameter is declared with the enum type when the SDK has it.
SDK_ENUMS = {
    'MTLPrimitiveType', 'MTLIndexType', 'MTLCullMode', 'MTLWinding', 'MTLTriangleFillMode', 'MTLDepthClipMode',
    'MTLVisibilityResultMode', 'MTLStoreAction', 'MTLStoreActionOptions', 'MTLResourceUsage', 'MTLRenderStages',
    'MTLBarrierScope', 'MTLBlitOption', 'MTLDispatchType',
}


def fail(message):
    raise SystemExit('FAIL: ' + message)


def camel(name):
    head, *rest = name.split('_')
    return head + ''.join(p[:1].upper() + p[1:] for p in rest)


def wire_name(name):
    """The generated C++ member name of a schema field (wiregen.py: lower camel case, `gl` and `io` kept)."""
    return camel(name)


def split_encoding(types):
    """Return type and argument encodings of an Objective-C method type encoding (offsets removed)."""
    out, i = [], 0
    while i < len(types):
        start = i
        while types[i] in 'rnNoORV':
            i += 1
        if types[i] == '^':
            i += 1
            continue_from = i
            i = skip_type(types, continue_from)
        else:
            i = skip_type(types, i)
        out.append(types[start:i])
        while i < len(types) and types[i].isdigit():
            i += 1
    return out[0], out[3:]


def skip_type(types, i):
    if types[i] in '{(':
        close = '}' if types[i] == '{' else ')'
        depth = 0
        while True:
            if types[i] in '{(':
                depth += 1
            elif types[i] in '})':
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
    if types[i] == '@' and i + 1 < len(types) and types[i + 1] == '?':
        return i + 2
    return i + 1


def parse_field(text):
    m = re.match(r'^(\w+): ([^ =(]+(?: ?[^ =(]*)?)(?: \((\w+)\))?(?: = (\S+))?$', text.strip())
    if not m:
        fail(f'cannot parse field {text!r}')
    return {'name': m.group(1), 'type': m.group(2).strip(), 'meta': m.group(3), 'count': m.group(4)}


def driver_selectors(header):
    out = {}
    for m in re.finditer(r'inline constexpr DriverSelector k(\w+)Driver\[\] = \{\n(.*?)\n\};', header, re.S):
        out[m.group(1)] = {sel: types for kind, sel, types in
                           re.findall(r'\{SelectorKind::(\w+), "([^"]+)", "([^"]+)"', m.group(2)) if kind == 'Instance'}
    return out


class Method:
    def __init__(self, cls, selector, types, op):
        self.cls, self.selector, self.types, self.op = cls, selector, types, op
        self.ret, self.args = split_encoding(types)
        self.fields = [parse_field(a) for a in op['args']]
        if self.ret != 'v':
            fail(f'{selector}: returns {self.ret}')
        if len(self.args) != len(self.fields):
            fail(f'{selector}: {len(self.args)} arguments, op {op["name"]} has {len(self.fields)} fields')
        self.params = []  # (ObjC type, name)
        self.body = []
        self.keep = []    # vectors that must outlive the encode call

    def objc_type(self, field, encoding):
        t = field['type']
        if t.endswith('?') and t[:-1] in KIND_PROTOCOLS or t in KIND_PROTOCOLS:
            if encoding != '@':
                fail(f'{self.selector}: {field["name"]} is {t} with encoding {encoding}')
            return f'id<{KIND_PROTOCOLS[t.rstrip("?")]}>'
        if t == 'str?':
            return 'NSString *'
        if t in SCALARS:
            code, ctype = SCALARS[t]
            if encoding != code:
                fail(f'{self.selector}: {field["name"]} is {t} with encoding {encoding}')
            if field['meta'] in SDK_ENUMS:
                return field['meta']
            return ctype
        if t in STRUCTS:
            mtype, enc, _ = STRUCTS[t]
            if encoding != enc:
                fail(f'{self.selector}: {field["name"]} is {t} with encoding {encoding}')
            return mtype
        if t == 'Region':
            if encoding != REGION[1]:
                fail(f'{self.selector}: {field["name"]} is Region with encoding {encoding}')
            return REGION[0]
        if t == 'bytes':
            if encoding != 'r^v':
                fail(f'{self.selector}: {field["name"]} is bytes with encoding {encoding}')
            return 'const void *'
        m = re.match(r'^\[(\w+)(\??)\]$', t)
        if m:
            inner = m.group(1)
            if inner in KIND_PROTOCOLS:
                if encoding != 'r^@':
                    fail(f'{self.selector}: {field["name"]} is {t} with encoding {encoding}')
                return f'const id<{KIND_PROTOCOLS[inner]}> __unsafe_unretained *'
            if inner in ('u64',):
                if encoding != 'r^Q':
                    fail(f'{self.selector}: {field["name"]} is {t} with encoding {encoding}')
                return 'const NSUInteger *'
            if inner == 'f32':
                if encoding != 'r^f':
                    fail(f'{self.selector}: {field["name"]} is {t} with encoding {encoding}')
                return 'const float *'
            if inner in STRUCTS:
                if encoding != 'r^' + STRUCTS[inner][1]:
                    fail(f'{self.selector}: {field["name"]} is {t} with encoding {encoding}')
                return f'const {STRUCTS[inner][0]} *'
            if inner == 'Region':
                if encoding != 'r^' + REGION[1]:
                    fail(f'{self.selector}: {field["name"]} is {t} with encoding {encoding}')
                return f'const {REGION[0]} *'
        fail(f'{self.selector}: no conversion for {field["name"]}: {t} ({encoding})')

    def count_expr(self, expr):
        head, _, member = expr.partition('.')
        names = {f['name'] for f in self.fields}
        if head not in names:
            fail(f'{self.selector}: count {expr} names no field')
        return camel(head) + ('.' + camel(member) if member else '')

    def convert(self, field, var):
        t, w = field['type'], 'record.' + wire_name(field['name'])
        if t.rstrip('?') in KIND_PROTOCOLS:
            nil = t.endswith('?')
            self.body.append(f'    {w} = webmetal::EncoderReference(self, _cmd, *_encoder, {var}, {str(nil).lower()});')
        elif t == 'str?':
            self.body.append(f'    if (!webmetal::wire::strFromNSString({var}, {w}))')
            self.body.append('        WMLoudMethod(self, _cmd, "a string argument has no UTF-8 form");')
        elif t in SCALARS:
            self.body.append(f'    {w} = {var};')
        elif t in STRUCTS:
            members = ', '.join(f'{var}.{m}' for _, m in STRUCTS[t][2])
            self.body.append(f'    {w} = {{{members}}};')
        elif t == 'Region':
            self.body.append(f'    {w} = {{{{{var}.origin.x, {var}.origin.y, {var}.origin.z}}, '
                             f'{{{var}.size.width, {var}.size.height, {var}.size.depth}}}};')
        elif t == 'bytes':
            count = self.count_expr(field['count'])
            self.body.append(f'    {w} = {{static_cast<const uint8_t *>({var}), (size_t)({count})}};')
        else:
            m = re.match(r'^\[(\w+)(\??)\]$', t)
            inner = m.group(1)
            count = self.count_expr(field['count'])
            if inner in KIND_PROTOCOLS:
                vec = f'{camel(field["name"])}Handles'
                nil = 'true' if m.group(2) else 'false'
                self.body.append(f'    std::vector<uint64_t> {vec};')
                self.body.append(f'    for (NSUInteger i = 0; i < (NSUInteger)({count}); i++)')
                self.body.append(f'        {vec}.push_back(webmetal::EncoderReference(self, _cmd, *_encoder, {var}[i], {nil}));')
                self.body.append(f'    {w} = {{{vec}.data(), {vec}.size()}};')
            elif inner == 'u64':
                self.body.append(f'    {w} = {{reinterpret_cast<const uint64_t *>({var}), (size_t)({count})}};')
            elif inner == 'f32':
                self.body.append(f'    {w} = {{{var}, (size_t)({count})}};')
            else:
                wtype = inner
                vec = f'{camel(field["name"])}Wire'
                self.body.append(f'    std::vector<webmetal::wire::{wtype}> {vec};')
                self.body.append(f'    for (NSUInteger i = 0; i < (NSUInteger)({count}); i++)')
                if inner == 'Region':
                    self.body.append(f'        {vec}.push_back({{{{{var}[i].origin.x, {var}[i].origin.y, {var}[i].origin.z}}, '
                                     f'{{{var}[i].size.width, {var}[i].size.height, {var}[i].size.depth}}}});')
                else:
                    members = ', '.join(f'{var}[i].{mm}' for _, mm in STRUCTS[inner][2])
                    self.body.append(f'        {vec}.push_back({{{members}}});')
                self.body.append(f'    {w} = {{{vec}.data(), {vec}.size()}};')

    def source(self):
        pieces = self.selector.split(':')[:-1]
        decl = []
        for i, (field, encoding) in enumerate(zip(self.fields, self.args)):
            ctype = self.objc_type(field, encoding)
            var = camel(field['name'])
            decl.append(f'{pieces[i]}:({ctype}){var}')
            self.convert(field, var)
        head = f'- (void){" ".join(decl)}' if decl else f'- (void){self.selector}'
        out = [f'// {self.op["name"]} (0x{self.op["code"]:04x}).', head, '{', f'    webmetal::wire::{self.op["name"]} record;']
        out += self.body
        out += ['    webmetal::EmitEncoderRecord(self, _cmd, *_encoder, record);', '}', '']
        return '\n'.join(out)


def generate():
    schema = tomllib.loads(SCHEMA.read_text())
    selectors = driver_selectors(SURFACE.read_text())
    ops = {}
    for op in schema['op']:
        if op.get('channel') != 'stream':
            continue
        for meth in op.get('methods', []):
            protocol, selector = meth.split(' ', 1)
            ops.setdefault(selector, []).append((protocol, op))
    out = ['// Generated by guest/Commands/gen/encodergen.py from wire/schema/webmetal.toml and',
           '// guest/Generated/WebMetalTargetSurface.h. Do not edit: run the script.', '//',
           '// The encoder commands that write one stream record each, with every argument carried verbatim.', '',
           '#import "WebMetalCommandState.h"', '#import "../Plugin/WebMetalReport.h"', '',
           '#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"', '']
    counts = []
    for plugin, driver, protocols in ROLES:
        if driver not in selectors:
            fail(f'{driver} has no driver selectors in {SURFACE}')
        methods = []
        for selector, types in sorted(selectors[driver].items()):
            if selector in HAND_WRITTEN:
                continue
            matches = [op for protocol, op in ops.get(selector, []) if protocol in protocols]
            names = {op['name'] for op in matches}
            if not matches:
                continue
            if len(names) != 1:
                fail(f'{driver} {selector}: several ops {sorted(names)}')
            methods.append(Method(plugin, selector, types, matches[0]))
        out += [f'@implementation {plugin} (WebMetalRecords)', '']
        out += [m.source() for m in methods]
        out += ['@end', '']
        counts.append((plugin, len(methods)))
    return '\n'.join(out), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text, counts = generate()
    for plugin, n in counts:
        print(f'{plugin}: {n} record methods')
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            fail(f'{OUTPUT.relative_to(ROOT)} is stale; run {Path(__file__).relative_to(ROOT)}')
        print('PASS: generated encoder methods are current')
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    print(f'wrote {OUTPUT.relative_to(ROOT)}')


main()
