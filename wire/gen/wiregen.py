#!/usr/bin/env python3
"""Generate the WebMetal wire codecs from wire/schema/webmetal.toml.

Outputs (all committed; never edit them by hand):

  wire/src/generated.rs            Rust types, encoders, decoders, fuzz generators and op metadata
  wire/generated/WebMetalWire.h    C++17 / ObjC++ header for the guest: encoders for requests and stream records,
                                   decoders for replies and events
  wire/generated/WebMetalWireFuzz.h  C++ fuzz generators and comparisons used by the cross-language tests
  wire/generated/opcodes.md        the opcode table

`--check` regenerates in memory and exits 1 if any committed output differs.

The schema's SHA-256 is the protocol version (SCHEMA_HASH / kSchemaHash). The format rules are in wire/README.md and
wire/src/codec.rs; this file only maps schema types onto them. Fuzz generators draw from splitmix64 in the same order in
both languages, so one seed yields the same records in Rust and C++.
"""
import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path

WIRE = Path(__file__).resolve().parents[1]
SCHEMA = WIRE / 'schema/webmetal.toml'
OUT_RS = WIRE / 'src/generated.rs'
OUT_H = WIRE / 'generated/WebMetalWire.h'
OUT_FUZZ_H = WIRE / 'generated/WebMetalWireFuzz.h'
OUT_MD = WIRE / 'generated/opcodes.md'

FIELD = re.compile(r'^(\w+): (.+?)(?: \((\w+)\))?(?: = (\w+(?:\.\w+)?))?$')
SCALARS = {
    # name: (rust type, c++ type, rust codec method, c++ codec method)
    'u8': ('u8', 'uint8_t', 'u8', 'u8'),
    'u16': ('u16', 'uint16_t', 'u16', 'u16'),
    'u32': ('u32', 'uint32_t', 'u32', 'u32'),
    'u64': ('u64', 'uint64_t', 'u64', 'u64'),
    'i32': ('i32', 'int32_t', 'i32', 'i32'),
    'i64': ('i64', 'int64_t', 'i64', 'i64'),
    'f32': ('f32', 'float', 'f32', 'f32'),
    'f64': ('f64', 'double', 'f64', 'f64'),
    'bool': ('bool', 'bool', 'bool', 'boolean'),
}
RUST_KEYWORDS = {'type', 'from', 'ref', 'move', 'match', 'fn', 'impl', 'struct', 'enum', 'mod', 'use', 'as', 'in', 'loop',
                 'where', 'self', 'super', 'crate', 'static', 'const', 'let', 'mut', 'trait', 'unsafe', 'yield', 'box'}
CPP_KEYWORDS = {'default', 'delete', 'new', 'class', 'union', 'template', 'operator', 'register', 'auto', 'case', 'char',
                'const', 'continue', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 'long',
                'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'unsigned', 'void',
                'volatile', 'while', 'bool', 'true', 'false', 'this', 'virtual', 'private', 'public', 'protected',
                'friend', 'inline', 'namespace', 'using', 'try', 'catch', 'throw', 'explicit', 'export', 'mutable',
                'typename', 'id', 'nil', 'self', 'super', 'in', 'out', 'inout', 'oneway', 'bycopy', 'byref'}
CHANNELS = ('control', 'stream', 'event')


def camel(snake):
    parts = snake.split('_')
    return parts[0] + ''.join(p[:1].upper() + p[1:] for p in parts[1:])


def snake(name):
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def rust_ident(name):
    return 'r#' + name if name in RUST_KEYWORDS else name


def cpp_ident(name):
    c = camel(name)
    return c + '_' if c in CPP_KEYWORDS else c


# Type model -----------------------------------------------------------------------------------------------------------

class Type:
    """A schema type. kind is one of: scalar flag guest_addr surface handle struct opt_struct str opt_str opt_cstr bytes
    array fixed opt_value."""

    def __init__(self, kind, name=None, elem=None, n=None, nullable=False):
        self.kind, self.name, self.elem, self.n, self.nullable = kind, name, elem, n, nullable

    @property
    def variable(self):
        return self.kind in ('str', 'opt_str', 'opt_cstr', 'bytes', 'array', 'opt_value')

    def __repr__(self):
        return '%s(%s)' % (self.kind, self.name or self.elem or self.n or '')


class Field:
    def __init__(self, text, schema):
        m = FIELD.match(text)
        if not m:
            raise SystemExit('malformed field %r' % text)
        self.name, self.type_text, self.metal, self.bound = m.groups()
        self.type = schema.parse_type(self.type_text)
        if self.bound and not self.type.variable:
            raise SystemExit('constraint on a fixed-size field %r' % text)
        self.rust = rust_ident(self.name)
        self.cpp = cpp_ident(self.name)


class Struct:
    def __init__(self, d, schema):
        self.name = d['name']
        self.doc = d.get('doc')
        self.metal = d.get('metal') or d.get('class')
        self.is_descriptor = 'class' in d
        self.dictionary_entry = d.get('dictionary_entry', False)
        self.raw_fields = d['fields']
        self.fields = []
        self.schema = schema

    def resolve(self):
        self.fields = [Field(f, self.schema) for f in self.raw_fields]


class Op:
    def __init__(self, d, schema):
        self.name = d['name']
        self.code = d['code']
        self.channel = d['channel']
        if self.channel not in CHANNELS:
            raise SystemExit('%s: unknown channel %s' % (self.name, self.channel))
        self.methods = d.get('methods', [])
        self.doc = d.get('doc')
        self.receiver = d.get('receiver')
        self.result = d.get('result')
        self.fields = [Field(f, schema) for f in d.get('args', [])]
        self.reply = [Field(f, schema) for f in d.get('reply', [])] if self.channel == 'control' else None
        if d.get('reply') and self.channel != 'control':
            raise SystemExit('%s: only control requests have replies' % self.name)
        names = {f.name for f in self.fields}
        for f in self.fields:
            if f.bound:
                head = f.bound.split('.')[0]
                if head not in names:
                    raise SystemExit('%s: %s bound to unknown field %s' % (self.name, f.name, f.bound))
        self.rust_snake = snake(self.name)


class Schema:
    def __init__(self, path):
        self.raw = path.read_bytes()
        self.hash = hashlib.sha256(self.raw).digest()
        data = tomllib.loads(self.raw.decode('utf-8'))
        self.data = data
        # The codec runtimes (wire/src/codec.rs, wire/src/value.rs, the C++ runtime below) implement these values.
        expected = {'record_alignment': 8, 'header': ['opcode: u16', 'flags: u16', 'length: u32'], 'reply_bit': 0x8000,
                    'value_depth_limit': 32}
        for key, value in expected.items():
            if data.get('schema', {}).get(key) != value:
                raise SystemExit('schema.%s must be %r: the codec runtimes implement that value' % (key, value))
        self.objects = {o['name']: o for o in data.get('object', [])}
        self.structs = {}
        for d in data.get('struct', []):
            if d['name'] in self.structs:
                raise SystemExit('duplicate struct %s' % d['name'])
            self.structs[d['name']] = Struct(d, self)
        for s in self.structs.values():
            s.resolve()
        self.ops = [Op(d, self) for d in data.get('op', [])]
        seen = {}
        for op in self.ops:
            if op.code in seen:
                raise SystemExit('opcode 0x%04x used twice' % op.code)
            if op.code & 0x8000 or op.code == 0:
                raise SystemExit('%s: opcode 0x%04x has the reply bit set or is zero' % (op.name, op.code))
            seen[op.code] = op
            if op.name in self.structs or op.name + 'Reply' in self.structs:
                raise SystemExit('op %s collides with a struct name' % op.name)
        self.by_channel = {c: [op for op in self.ops if op.channel == c] for c in CHANNELS}
        for op in self.ops:
            if (op.channel == 'event' and any(f.bound for f in op.fields)) or any(f.bound for f in (op.reply or [])):
                raise SystemExit('%s: length constraints are only supported guest-to-host' % op.name)
        self.order_structs()
        self.directions()

    def parse_type(self, text):
        if text in SCALARS:
            return Type('scalar', text)
        if text in ('block', 'out', 'present'):
            return Type('flag', text)
        if text == 'guest_addr':
            return Type('guest_addr')
        if text == 'surface':
            return Type('surface')
        if text == 'str':
            return Type('str')
        if text in ('str?', 'url?'):
            return Type('opt_str')
        if text == 'cstr?':
            return Type('opt_cstr')
        if text == 'bytes':
            return Type('bytes')
        if text == 'value?':
            return Type('opt_value')
        m = re.match(r'^\[u8; (\d+)\]$', text)
        if m:
            return Type('fixed', n=int(m.group(1)))
        m = re.match(r'^\[(.+)\]$', text)
        if m:
            elem = self.parse_type(m.group(1))
            if elem.variable and elem.kind != 'str':
                raise SystemExit('array of variable elements %r' % text)
            return Type('array', elem=elem)
        base, nullable = (text[:-1], True) if text.endswith('?') else (text, False)
        if base in self.objects:
            return Type('handle', base, nullable=nullable)
        if base in self.structs:
            return Type('opt_struct' if nullable else 'struct', base)
        raise SystemExit('unknown type %r' % text)

    def order_structs(self):
        """Structs in dependency order (C++ needs a definition before use)."""
        order, state = [], {}

        def visit(name):
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                raise SystemExit('recursive struct %s' % name)
            state[name] = 1
            for f in self.structs[name].fields:
                for t in (f.type, f.type.elem):
                    if t is not None and t.kind in ('struct', 'opt_struct'):
                        visit(t.name)
            state[name] = 2
            order.append(self.structs[name])
        for name in self.structs:
            visit(name)
        self.struct_order = order

    def directions(self):
        """Which structs travel guest-to-host (request and stream fields) and host-to-guest (replies and events)."""
        def reach(fields, out):
            for f in fields:
                for t in (f.type, f.type.elem):
                    if t is not None and t.kind in ('struct', 'opt_struct') and t.name not in out:
                        out.add(t.name)
                        reach(self.structs[t.name].fields, out)
        self.to_host, self.to_guest = set(), set()
        for op in self.ops:
            if op.channel in ('control', 'stream'):
                reach(op.fields, self.to_host)
            if op.channel == 'control':
                reach(op.reply, self.to_guest)
            if op.channel == 'event':
                reach(op.fields, self.to_guest)
        self.to_guest.add('Failure')
        reach(self.structs['Failure'].fields, self.to_guest)
        for name in self.to_host & self.to_guest:
            if any(f.type.variable or f.type.kind == 'opt_struct' for f in self.structs[name].fields):
                raise SystemExit('struct %s with variable fields is used in both directions' % name)
        for name in self.to_guest:
            for f in self.structs[name].fields:
                if f.type.kind in ('opt_value', 'opt_cstr'):
                    raise SystemExit('%s.%s: %s is not supported host-to-guest' % (name, f.name, f.type.kind))


def bound_groups(fields):
    """Distinct length bounds in order of first use, with whether any field bound to them is a byte string."""
    out = []
    for f in fields:
        if f.bound:
            for b in out:
                if b[0] == f.bound:
                    b[1] |= f.type.kind in ('bytes', 'str', 'opt_str', 'opt_cstr')
                    break
            else:
                out.append([f.bound, f.type.kind in ('bytes', 'str', 'opt_str', 'opt_cstr')])
    return out


def doc_lines(text, prefix):
    if not text:
        return []
    out, line = [], prefix
    for word in text.split():
        if len(line) + len(word) + 1 > 118 and line.strip() != prefix.strip():
            out.append(line.rstrip())
            line = prefix
        line += word + ' '
    out.append(line.rstrip())
    return out


def field_doc(f):
    parts = []
    if f.metal:
        parts.append('`%s`' % f.metal)
    parts.append('wire type `%s`' % f.type_text)
    if f.bound:
        parts.append('length equals `%s`' % f.bound)
    return ', '.join(parts) + '.'


# Rust ----------------------------------------------------------------------------------------------------------------

def rust_type(t):
    k = t.kind
    if k == 'scalar':
        return SCALARS[t.name][0]
    if k == 'flag':
        return 'bool'
    if k == 'guest_addr':
        return 'GuestAddr'
    if k == 'surface':
        return 'SurfaceId'
    if k == 'handle':
        return 'Option<Handle>' if t.nullable else 'Handle'
    if k == 'struct':
        return t.name
    if k == 'opt_struct':
        return 'Option<%s>' % t.name
    if k == 'str':
        return 'String'
    if k == 'opt_str':
        return 'Option<String>'
    if k == 'opt_cstr':
        return 'Option<Vec<u8>>'
    if k == 'bytes':
        return 'Vec<u8>'
    if k == 'array':
        return 'Vec<%s>' % rust_type(t.elem)
    if k == 'fixed':
        return '[u8; %d]' % t.n
    if k == 'opt_value':
        return 'Option<Value>'
    raise AssertionError(k)


def rust_encode(t, x, name, ind):
    """Statements encoding place expression x (an owned place, not a reference)."""
    k = t.kind
    p = ' ' * ind
    if k == 'scalar':
        return [p + 'w.%s(%s);' % (SCALARS[t.name][2], x)]
    if k == 'flag':
        return [p + 'w.bool(%s);' % x]
    if k == 'guest_addr':
        return [p + 'w.u64(%s.0);' % x]
    if k == 'surface':
        return [p + 'w.u32(%s.0);' % x]
    if k == 'handle':
        return [p + ('w.opt_handle(%s);' if t.nullable else 'w.handle(%s);') % x]
    if k == 'struct':
        return [p + '%s.encode_fields(w)?;' % x]
    if k == 'opt_struct':
        return [p + 'match &%s {' % x, p + '    None => w.u8(0),', p + '    Some(v) => {', p + '        w.u8(1);',
                p + '        v.encode_fields(w)?;', p + '    }', p + '}']
    if k == 'str':
        return [p + 'w.str(&%s, "%s")?;' % (x, name)]
    if k == 'opt_str':
        return [p + 'w.opt_str(%s.as_deref(), "%s")?;' % (x, name)]
    if k == 'opt_cstr':
        return [p + 'w.opt_cstr(%s.as_deref(), "%s")?;' % (x, name)]
    if k == 'bytes':
        return [p + 'w.bytes(&%s, "%s")?;' % (x, name)]
    if k == 'fixed':
        return [p + 'w.raw(&%s);' % x]
    if k == 'opt_value':
        return [p + 'crate::value::encode_opt_value(w, %s.as_ref())?;' % x]
    if k == 'array':
        copy = t.elem.kind in ('scalar', 'flag', 'handle', 'guest_addr', 'surface')
        return ([p + 'w.count(%s.len(), "%s")?;' % (x, name), p + 'for %s in &%s {' % ('&e' if copy else 'e', x)]
                + rust_encode(t.elem, 'e', name, ind + 4) + [p + '}'])
    raise AssertionError(k)


def rust_decode(t, name):
    k = t.kind
    if k == 'scalar':
        return 'r.%s()?' % SCALARS[t.name][2]
    if k == 'flag':
        return 'r.bool()?'
    if k == 'guest_addr':
        return 'GuestAddr(r.u64()?)'
    if k == 'surface':
        return 'SurfaceId(r.u32()?)'
    if k == 'handle':
        return 'r.opt_handle()?' if t.nullable else 'r.handle("%s")?' % name
    if k == 'struct':
        return '%s::decode_fields(r)?' % t.name
    if k == 'opt_struct':
        return 'if r.bool()? { Some(%s::decode_fields(r)?) } else { None }' % t.name
    if k == 'str':
        return 'r.str("%s")?' % name
    if k == 'opt_str':
        return 'r.opt_str("%s")?' % name
    if k == 'opt_cstr':
        return 'r.opt_cstr("%s")?' % name
    if k == 'bytes':
        return 'r.bytes("%s")?' % name
    if k == 'fixed':
        return 'r.raw(%d)?.try_into().unwrap()' % t.n
    if k == 'opt_value':
        return 'crate::value::decode_opt_value(r)?'
    if k == 'array':
        return ('{ let n = r.count("%s")?; let mut v = Vec::with_capacity(n); for _ in 0..n { v.push(%s); } v }'
                % (name, rust_decode(t.elem, name)))
    raise AssertionError(k)


def rust_random(t, length=None):
    k = t.kind
    if k == 'scalar':
        return 'rng.%s()' % ('boolean' if t.name == 'bool' else t.name)
    if k == 'flag':
        return 'rng.boolean()'
    if k == 'guest_addr':
        return 'GuestAddr(rng.u64())'
    if k == 'surface':
        return 'SurfaceId(rng.u32())'
    if k == 'handle':
        return 'rng.opt_handle()' if t.nullable else 'rng.handle()'
    if k == 'struct':
        return '%s::random(rng)' % t.name
    if k == 'opt_struct':
        return 'if rng.boolean() { Some(%s::random(rng)) } else { None }' % t.name
    if k == 'fixed':
        return 'rng.fixed::<%d>()' % t.n
    n = length or {'bytes': 'rng.bytes_len()', 'array': 'rng.array_len()'}.get(k)
    if k == 'str':
        return 'rng.string(%s)' % length if length else '{ let n = rng.string_len(); rng.string(n) }'
    if k == 'opt_str':
        return 'rng.opt_string()' if length is None else 'Some(rng.string(%s))' % length
    if k == 'opt_cstr':
        return 'rng.opt_cstr()' if length is None else 'Some(rng.cstr(%s))' % length
    if k == 'bytes':
        return '{ let n = %s; rng.bytes(n) }' % n
    if k == 'opt_value':
        return 'rng.opt_value()'
    if k == 'array':
        return '{ let n = %s; let mut v = Vec::with_capacity(n as usize); for _ in 0..n { v.push(%s); } v }' % (
            n, rust_random(t.elem))
    raise AssertionError(k)


def rust_bound_expr(bound, prefix):
    head, _, member = bound.partition('.')
    return '%s%s%s' % (prefix, rust_ident(head), '.' + rust_ident(member) if member else '')


def rust_len(t, x):
    if t.kind in ('opt_str', 'opt_cstr'):
        return '%s.as_ref().map_or(0, |s| s.len())' % x
    return '%s.len()' % x


def rust_record_body(name, fields, what):
    """encode_fields / decode_fields / check / random for a struct or op payload."""
    out = []
    bounded = any(f.bound for f in fields)
    used_w = bool(fields)
    out.append('    fn encode_fields(&self, %s: &mut Writer<\'_>) -> Result<(), EncodeError> {' % ('w' if used_w else '_w'))
    if what == 'op' and bounded:
        out.append('        self.check().map_err(EncodeError::Constraint)?;')
    if what == 'failure':
        out.append('        if !crate::failure_valid(self) {')
        out.append('            return Err(EncodeError::BadFailure);')
        out.append('        }')
    for f in fields:
        out += rust_encode(f.type, 'self.' + f.rust, f.name, 8)
    out.append('        Ok(())')
    out.append('    }')
    out.append('')
    used_r = bool(fields)
    out.append('    fn decode_fields(%s: &mut Reader<\'_>) -> Result<Self, DecodeError> {' % ('r' if used_r else '_r'))
    for f in fields:
        out.append('        let %s = %s;' % (f.rust, rust_decode(f.type, f.name)))
    out.append('        let v = %s { %s };' % (name, ', '.join(f.rust for f in fields)))
    if what == 'op' and bounded:
        out.append('        v.check().map_err(|field| r.error(DecodeErrorKind::Constraint(field)))?;')
    if what == 'failure':
        out.append('        if !crate::failure_valid(&v) {')
        out.append('            return Err(r.error(DecodeErrorKind::BadFailure));')
        out.append('        }')
    out.append('        Ok(v)')
    out.append('    }')
    if what == 'op':
        out.append('')
        out.append('    /// Check the length constraints between fields; on failure names the constrained field.')
        out.append('    pub fn check(&self) -> Result<(), &\'static str> {')
        for f in fields:
            if f.bound:
                out.append('        if %s as u64 != %s {' % (rust_len(f.type, 'self.' + f.rust), rust_bound_expr(f.bound, 'self.')))
                out.append('            return Err("%s");' % f.name)
                out.append('        }')
        out.append('        Ok(())')
        out.append('    }')
    out.append('')
    out.append('    /// A random value, drawn in the same order as the C++ generator in WebMetalWireFuzz.h.')
    out.append('    pub fn random(%s: &mut Rng) -> Self {' % ('rng' if fields or what == 'failure' else '_rng'))
    if what == 'failure':
        out.append('        crate::fuzz::random_failure(rng)')
    elif not bound_groups(fields):
        for f in fields:
            out.append('        let %s = %s;' % (f.rust, rust_random(f.type)))
        out.append('        %s { %s }' % (name, ', '.join(f.rust for f in fields)))
    else:
        # Records with length constraints: fixed-size fields in order, then one length per bound, then the
        # variable-length fields in order (the C++ generator follows the same three phases).
        groups = bound_groups(fields)
        whole = {g[0] for g in groups if '.' not in g[0]}
        member_heads = {g[0].split('.')[0] for g in groups if '.' in g[0]}
        for f in fields:
            if f.type.variable:
                continue
            if f.name in whole:
                out.append('        %s; // drawn to keep the C++ draw order, then replaced by the bound length' % rust_random(f.type))
            else:
                out.append('        let %s%s = %s;' % ('mut ' if f.name in member_heads else '', f.rust, rust_random(f.type)))
        for i, (bound, bytes_like) in enumerate(groups):
            out.append('        let len%d = rng.%s();' % (i, 'bytes_len' if bytes_like else 'array_len'))
            if '.' in bound:
                out.append('        %s = len%d;' % (rust_bound_expr(bound, ''), i))
            else:
                out.append('        let %s = len%d;' % (rust_ident(bound), i))
        for f in fields:
            if f.type.variable:
                length = None
                if f.bound:
                    length = 'len%d' % [g[0] for g in groups].index(f.bound)
                out.append('        let %s = %s;' % (f.rust, rust_random(f.type, length)))
        out.append('        %s { %s }' % (name, ', '.join(f.rust for f in fields)))
    out.append('    }')
    return out


def rust_struct_def(name, fields, docs, derive='Clone, Debug, PartialEq'):
    out = docs + ['#[derive(%s)]' % derive, 'pub struct %s {' % name]
    for f in fields:
        out.append('    /// ' + field_doc(f))
        out.append('    pub %s: %s,' % (f.rust, rust_type(f.type)))
    out.append('}')
    return out


def gen_rust(schema):
    o = []
    o.append('// Generated by wire/gen/wiregen.py from wire/schema/webmetal.toml. Do not edit.')
    o.append('//')
    o.append('// Schema SHA-256: %s' % schema.hash.hex())
    o.append('')
    o.append('#![allow(clippy::all)]')
    o.append('')
    o.append('use crate::codec::{split_record, DecodeError, DecodeErrorKind, EncodeError, GuestAddr, Handle, Reader, SurfaceId, Writer, REPLY_BIT};')
    o.append('use crate::fuzz::Rng;')
    o.append('use crate::value::Value;')
    o.append('')
    o.append('/// SHA-256 of wire/schema/webmetal.toml: the protocol version exchanged in [`Open`].')
    o.append('pub const SCHEMA_HASH: [u8; 32] = [%s];' % ', '.join('0x%02x' % b for b in schema.hash))
    o.append('')
    o.append('/// Opcode values.')
    o.append('pub mod opcode {')
    for op in schema.ops:
        o.append('    pub const %s: u16 = 0x%04x;' % (snake(op.name).upper(), op.code))
    o.append('}')
    o.append('')
    # structs
    for s in schema.struct_order:
        docs = []
        if s.metal:
            docs.append('/// Wire form of `%s`.' % s.metal)
        docs += doc_lines(s.doc, '/// ')
        o += rust_struct_def(s.name, s.fields, docs)
        o.append('')
        o.append('impl %s {' % s.name)
        o += rust_record_body(s.name, s.fields, 'failure' if s.name == 'Failure' else 'struct')
        o.append('}')
        o.append('')
    # ops
    for op in schema.ops:
        docs = ['/// `0x%04x` %s record.' % (op.code, op.channel)]
        if op.methods:
            docs.append('///')
            docs.append('/// Metal methods: %s.' % ', '.join('`-[%s %s]`' % tuple(m.split(' ')) for m in op.methods))
        if op.receiver:
            docs.append('/// `%s` is the receiver.' % op.receiver)
        if op.result:
            docs.append('/// `%s` is the handle the guest allocated for the new object.' % op.result)
        if op.doc:
            docs.append('///')
            docs += doc_lines(op.doc, '/// ')
        o += rust_struct_def(op.name, op.fields, docs)
        o.append('')
        o.append('impl %s {' % op.name)
        o.append('    pub const OPCODE: u16 = 0x%04x;' % op.code)
        o.append('')
        o += rust_record_body(op.name, op.fields, 'op')
        o.append('}')
        o.append('')
        if op.channel == 'control':
            rname = op.name + 'Reply'
            o += rust_struct_def(rname, op.reply, ['/// Reply fields of [`%s`].' % op.name])
            o.append('')
            o.append('impl %s {' % rname)
            o += rust_record_body(rname, op.reply, 'struct')
            o.append('}')
            o.append('')
    # channel enums
    for channel, enum, doc in (('stream', 'Command', 'A command-stream record.'),
                               ('control', 'Request', 'A control request.'),
                               ('event', 'Event', 'An event record.')):
        ops = schema.by_channel[channel]
        o.append('/// %s' % doc)
        o.append('#[derive(Clone, Debug, PartialEq)]')
        o.append('pub enum %s {' % enum)
        for op in ops:
            o.append('    %s(%s),' % (op.name, op.name))
        o.append('}')
        o.append('')
        o.append('impl %s {' % enum)
        o.append('    /// The record\'s opcode.')
        o.append('    pub fn opcode(&self) -> u16 {')
        o.append('        match self {')
        for op in ops:
            o.append('            %s::%s(_) => 0x%04x,' % (enum, op.name, op.code))
        o.append('        }')
        o.append('    }')
        o.append('')
        o.append('    /// Append the record to `out`, whose length must be a multiple of 8.')
        o.append('    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), EncodeError> {')
        o.append('        let start = out.len();')
        o.append('        let mut w = Writer::begin(out, self.opcode());')
        o.append('        let r = match self {')
        for op in ops:
            o.append('            %s::%s(v) => v.encode_fields(&mut w),' % (enum, op.name))
        o.append('        };')
        o.append('        match r {')
        o.append('            Ok(()) => w.finish(),')
        o.append('            Err(e) => {')
        o.append('                drop(w);')
        o.append('                out.truncate(start);')
        o.append('                Err(e)')
        o.append('            }')
        o.append('        }')
        o.append('    }')
        o.append('')
        o.append('    /// Decode the record at the start of `bytes` (stream offset `base`); returns it and its length.')
        o.append('    pub fn decode(bytes: &[u8], base: usize) -> Result<(%s, usize), DecodeError> {' % enum)
        o.append('        let (opcode, record) = split_record(bytes, base)?;')
        o.append('        let mut r = Reader::new(record, base);')
        o.append('        let v = match opcode {')
        for op in ops:
            o.append('            0x%04x => %s::%s(%s::decode_fields(&mut r)?),' % (op.code, enum, op.name, op.name))
        o.append('            _ => return Err(DecodeError { offset: base, kind: unknown_or_wrong_channel(opcode) }),')
        o.append('        };')
        o.append('        r.finish()?;')
        o.append('        Ok((v, record.len()))')
        o.append('    }')
        o.append('')
        o.append('    /// Decode every record of `bytes` in order.')
        o.append('    pub fn decode_all(bytes: &[u8]) -> Result<Vec<(usize, %s)>, DecodeError> {' % enum)
        o.append('        let mut out = Vec::new();')
        o.append('        let mut at = 0;')
        o.append('        while at < bytes.len() {')
        o.append('            let (v, len) = %s::decode(&bytes[at..], at)?;' % enum)
        o.append('            out.push((at, v));')
        o.append('            at += len;')
        o.append('        }')
        o.append('        Ok(out)')
        o.append('    }')
        o.append('')
        o.append('    /// A random record of this channel (the op chosen uniformly in schema order).')
        o.append('    pub fn random(rng: &mut Rng) -> %s {' % enum)
        o.append('        match rng.below(%d) {' % len(ops))
        for i, op in enumerate(ops):
            o.append('            %d => %s::%s(%s::random(rng)),' % (i, enum, op.name, op.name))
        o.append('            _ => unreachable!(),')
        o.append('        }')
        o.append('    }')
        o.append('}')
        o.append('')
    # replies
    ctrl = schema.by_channel['control']
    o.append('/// A reply to a control request: the request\'s reply fields, or the failure that stopped it.')
    o.append('#[derive(Clone, Debug, PartialEq)]')
    o.append('pub enum Reply {')
    for op in ctrl:
        o.append('    %s(Result<%sReply, Failure>),' % (op.name, op.name))
    o.append('}')
    o.append('')
    o.append('impl Reply {')
    o.append('    /// The opcode of the request this replies to.')
    o.append('    pub fn request_opcode(&self) -> u16 {')
    o.append('        match self {')
    for op in ctrl:
        o.append('            Reply::%s(_) => 0x%04x,' % (op.name, op.code))
    o.append('        }')
    o.append('    }')
    o.append('')
    o.append('    /// Append the reply record (opcode = request opcode | 0x8000) to `out`.')
    o.append('    pub fn encode(&self, out: &mut Vec<u8>) -> Result<(), EncodeError> {')
    o.append('        let start = out.len();')
    o.append('        let mut w = Writer::begin(out, self.request_opcode() | REPLY_BIT);')
    o.append('        let r = match self {')
    for op in ctrl:
        o.append('            Reply::%s(v) => encode_outcome(&mut w, v, %sReply::encode_fields),' % (op.name, op.name))
    o.append('        };')
    o.append('        match r {')
    o.append('            Ok(()) => w.finish(),')
    o.append('            Err(e) => {')
    o.append('                drop(w);')
    o.append('                out.truncate(start);')
    o.append('                Err(e)')
    o.append('            }')
    o.append('        }')
    o.append('    }')
    o.append('')
    o.append('    /// Decode the reply record at the start of `bytes`; returns it and its length.')
    o.append('    pub fn decode(bytes: &[u8], base: usize) -> Result<(Reply, usize), DecodeError> {')
    o.append('        let (opcode, record) = split_record(bytes, base)?;')
    o.append('        if opcode & REPLY_BIT == 0 {')
    o.append('            return Err(DecodeError { offset: base, kind: DecodeErrorKind::WrongChannel(opcode) });')
    o.append('        }')
    o.append('        let mut r = Reader::new(record, base);')
    o.append('        let v = match opcode & !REPLY_BIT {')
    for op in ctrl:
        o.append('            0x%04x => Reply::%s(decode_outcome(&mut r, %sReply::decode_fields)?),' % (op.code, op.name, op.name))
    o.append('            _ => return Err(DecodeError { offset: base, kind: DecodeErrorKind::UnknownOpcode(opcode) }),')
    o.append('        };')
    o.append('        r.finish()?;')
    o.append('        Ok((v, record.len()))')
    o.append('    }')
    o.append('')
    o.append('    /// A random reply (request chosen uniformly in schema order, then success or failure).')
    o.append('    pub fn random(rng: &mut Rng) -> Reply {')
    o.append('        match rng.below(%d) {' % len(ctrl))
    for i, op in enumerate(ctrl):
        o.append('            %d => Reply::%s(if rng.below(4) == 0 { Err(crate::fuzz::random_error_failure(rng)) } else { Ok(%sReply::random(rng)) }),'
                 % (i, op.name, op.name))
    o.append('            _ => unreachable!(),')
    o.append('        }')
    o.append('    }')
    o.append('}')
    o.append('')
    o.append('fn encode_outcome<T>(w: &mut Writer<\'_>, v: &Result<T, Failure>, f: fn(&T, &mut Writer<\'_>) -> Result<(), EncodeError>) -> Result<(), EncodeError> {')
    o.append('    match v {')
    o.append('        Ok(body) => {')
    o.append('            w.u32(0);')
    o.append('            f(body, w)')
    o.append('        }')
    o.append('        Err(failure) => {')
    o.append('            if failure.status == 0 {')
    o.append('                return Err(EncodeError::BadFailure);')
    o.append('            }')
    o.append('            failure.encode_fields(w)')
    o.append('        }')
    o.append('    }')
    o.append('}')
    o.append('')
    o.append('fn decode_outcome<T>(r: &mut Reader<\'_>, f: fn(&mut Reader<\'_>) -> Result<T, DecodeError>) -> Result<Result<T, Failure>, DecodeError> {')
    o.append('    let start = r.position();')
    o.append('    if r.u32()? == 0 {')
    o.append('        return Ok(Ok(f(r)?));')
    o.append('    }')
    o.append('    r.set_position(start);')
    o.append('    Ok(Err(Failure::decode_fields(r)?))')
    o.append('}')
    o.append('')
    # unknown/wrong channel
    o.append('fn unknown_or_wrong_channel(opcode: u16) -> DecodeErrorKind {')
    o.append('    if opcode & REPLY_BIT != 0 || op_info(opcode).is_some() {')
    o.append('        DecodeErrorKind::WrongChannel(opcode)')
    o.append('    } else {')
    o.append('        DecodeErrorKind::UnknownOpcode(opcode)')
    o.append('    }')
    o.append('}')
    o.append('')
    # metadata
    o.append('/// Which direction and framing a record uses.')
    o.append('#[derive(Clone, Copy, Debug, PartialEq, Eq)]')
    o.append('pub enum Channel {')
    o.append('    /// Guest to host, one reply each.')
    o.append('    Control,')
    o.append('    /// Guest to host inside a command buffer\'s stream.')
    o.append('    Stream,')
    o.append('    /// Host to guest, collected with FetchEvents.')
    o.append('    Event,')
    o.append('}')
    o.append('')
    o.append('/// Static description of an op, for diagnostics (an unimplemented opcode is reported with its Metal methods).')
    o.append('#[derive(Clone, Copy, Debug)]')
    o.append('pub struct OpInfo {')
    o.append('    pub opcode: u16,')
    o.append('    pub name: &\'static str,')
    o.append('    pub channel: Channel,')
    o.append('    /// "Protocol selector" pairs this record carries.')
    o.append('    pub methods: &\'static [&\'static str],')
    o.append('}')
    o.append('')
    o.append('/// Every op, in schema order.')
    o.append('pub const OPS: &[OpInfo] = &[')
    for op in schema.ops:
        o.append('    OpInfo { opcode: 0x%04x, name: "%s", channel: Channel::%s, methods: &[%s] },' % (
            op.code, op.name, op.channel.capitalize(), ', '.join('"%s"' % m for m in op.methods)))
    o.append('];')
    o.append('')
    o.append('/// The op with this opcode.')
    o.append('pub fn op_info(opcode: u16) -> Option<&\'static OpInfo> {')
    o.append('    OPS.iter().find(|op| op.opcode == opcode)')
    o.append('}')
    return '\n'.join(o) + '\n'


# C++ -----------------------------------------------------------------------------------------------------------------

def cpp_ref_type(t):
    """Type of a field in a guest-to-host struct (non-owning views)."""
    k = t.kind
    if k == 'scalar':
        return SCALARS[t.name][1]
    if k == 'flag':
        return 'bool'
    if k in ('guest_addr', 'handle'):
        return 'uint64_t'
    if k == 'surface':
        return 'uint32_t'
    if k == 'struct':
        return t.name
    if k == 'opt_struct':
        return 'Opt<%s>' % t.name
    if k in ('str', 'opt_str', 'opt_cstr'):
        return 'Str'
    if k == 'bytes':
        return 'Bytes'
    if k == 'array':
        return 'Span<%s>' % cpp_ref_type(t.elem)
    if k == 'fixed':
        return 'std::array<uint8_t, %d>' % t.n
    if k == 'opt_value':
        return 'const Value*'
    raise AssertionError(k)


def cpp_own_type(t):
    """Type of a field in a host-to-guest struct (owning)."""
    k = t.kind
    if k in ('scalar', 'flag', 'guest_addr', 'handle', 'surface', 'struct', 'fixed'):
        return cpp_ref_type(t)
    if k == 'opt_struct':
        return 'std::optional<%s>' % t.name
    if k == 'str':
        return 'std::string'
    if k == 'opt_str':
        return 'std::optional<std::string>'
    if k == 'bytes':
        return 'std::vector<uint8_t>'
    if k == 'array':
        return 'std::vector<%s>' % cpp_own_type(t.elem)
    raise AssertionError(k)


def cpp_default(t):
    if t.kind in ('scalar', 'guest_addr', 'handle', 'surface'):
        return ' = 0' if not (t.kind == 'scalar' and t.name == 'bool') else ' = false'
    if t.kind == 'flag':
        return ' = false'
    if t.kind == 'opt_value':
        return ' = nullptr'
    return '{}'


def cpp_encode(t, x, name, ind):
    k = t.kind
    p = ' ' * ind
    if k == 'scalar':
        return [p + 'w.%s(%s);' % (SCALARS[t.name][3], x)]
    if k == 'flag':
        return [p + 'w.boolean(%s);' % x]
    if k == 'guest_addr':
        return [p + 'w.u64(%s);' % x]
    if k == 'surface':
        return [p + 'w.u32(%s);' % x]
    if k == 'handle':
        if t.nullable:
            return [p + 'w.u64(%s);' % x]
        return [p + 'if (%s == 0) return false;' % x, p + 'w.u64(%s);' % x]
    if k == 'struct':
        return [p + 'if (!encodeFields(w, %s)) return false;' % x]
    if k == 'opt_struct':
        return [p + 'w.boolean(%s.present);' % x, p + 'if (%s.present && !encodeFields(w, %s.value)) return false;' % (x, x)]
    if k == 'str':
        return [p + 'if (!%s.present || !w.str(%s)) return false;' % (x, x)]
    if k == 'opt_str':
        return [p + 'if (!w.optStr(%s)) return false;' % x]
    if k == 'opt_cstr':
        return [p + 'if (!w.optCstr(%s)) return false;' % x]
    if k == 'bytes':
        return [p + 'if (!w.bytes(%s)) return false;' % x]
    if k == 'fixed':
        return [p + 'w.raw(%s.data(), %d);' % (x, t.n)]
    if k == 'opt_value':
        return [p + 'if (!w.optValue(%s)) return false;' % x]
    if k == 'array':
        return ([p + 'if (%s.size != 0 && %s.data == nullptr) return false;' % (x, x),
                 p + 'if (!w.count(%s.size)) return false;' % x,
                 p + 'for (size_t i = 0; i < %s.size; ++i) {' % x]
                + cpp_encode(t.elem, '%s.data[i]' % x, name, ind + 4) + [p + '}'])
    raise AssertionError(k)


def cpp_decode(t, x, name, ind):
    """Statements decoding into owning place x."""
    k = t.kind
    p = ' ' * ind
    if k == 'scalar':
        return [p + 'if (!r.%s(%s)) return false;' % (SCALARS[t.name][3], x)]
    if k == 'flag':
        return [p + 'if (!r.boolean(%s)) return false;' % x]
    if k == 'guest_addr':
        return [p + 'if (!r.u64(%s)) return false;' % x]
    if k == 'surface':
        return [p + 'if (!r.u32(%s)) return false;' % x]
    if k == 'handle':
        out = [p + 'if (!r.u64(%s)) return false;' % x]
        if not t.nullable:
            out.append(p + 'if (%s == 0) return r.fail(DecodeError::NullHandle);' % x)
        return out
    if k == 'struct':
        return [p + 'if (!decodeFields(r, %s)) return false;' % x]
    if k == 'opt_struct':
        return [p + '{', p + '    bool present = false;', p + '    if (!r.boolean(present)) return false;',
                p + '    if (present) {', p + '        %s.emplace();' % x,
                p + '        if (!decodeFields(r, *%s)) return false;' % x, p + '    } else {', p + '        %s.reset();' % x,
                p + '    }', p + '}']
    if k == 'str':
        return [p + 'if (!r.str(%s)) return false;' % x]
    if k == 'opt_str':
        return [p + 'if (!r.optStr(%s)) return false;' % x]
    if k == 'bytes':
        return [p + 'if (!r.bytes(%s)) return false;' % x]
    if k == 'fixed':
        return [p + 'if (!r.raw(%s.data(), %d)) return false;' % (x, t.n)]
    if k == 'array':
        return ([p + '{', p + '    uint32_t n = 0;', p + '    if (!r.count(n)) return false;', p + '    %s.clear();' % x,
                 p + '    %s.resize(n);' % x, p + '    for (uint32_t i = 0; i < n; ++i) {']
                + cpp_decode(t.elem, '%s[i]' % x, name, ind + 8) + [p + '    }', p + '}'])
    raise AssertionError(k)


def cpp_len(t, x):
    if t.kind in ('str', 'opt_str', 'opt_cstr', 'bytes', 'array'):
        return '%s.size' % x
    raise AssertionError(t.kind)


def cpp_bound_expr(bound, prefix):
    head, _, member = bound.partition('.')
    return '%s%s%s' % (prefix, cpp_ident(head), '.' + cpp_ident(member) if member else '')


def gen_cpp(schema):
    h = []
    h.append('// Generated by wire/gen/wiregen.py from wire/schema/webmetal.toml. Do not edit.')
    h.append('//')
    h.append('// WebMetal wire protocol for the guest: encoders for control requests and command-stream records, decoders for')
    h.append('// replies and events. C++17; under Objective-C++ it adds conversions from Foundation objects. The format is')
    h.append('// described in wire/README.md. Every encode function returns false, and leaves the writer unchanged, when the')
    h.append('// value breaks a rule of the format (a nil handle where none is allowed, a length constraint, an invalid string,')
    h.append('// an oversized record); the caller must then report the failure, never send a substitute.')
    h.append('//')
    h.append('// Schema SHA-256: %s' % schema.hash.hex())
    h.append('')
    h.append('#pragma once')
    h.append('')
    h.append(CPP_RUNTIME)
    h.append('constexpr std::array<uint8_t, 32> kSchemaHash = {%s};' % ', '.join('0x%02x' % b for b in schema.hash))
    h.append('')
    h.append('namespace opcode {')
    for op in schema.ops:
        h.append('constexpr uint16_t k%s = 0x%04x;' % (op.name, op.code))
    h.append('}  // namespace opcode')
    h.append('')
    # forward declarations of struct codecs
    for s in schema.struct_order:
        h += cpp_struct(schema, s)
    for op in schema.ops:
        if op.channel in ('control', 'stream'):
            h += cpp_g2h_op(op)
        else:
            h += cpp_h2g_record(op.name, op.fields, op.code, 'event', op.doc)
        if op.channel == 'control':
            h += cpp_h2g_record(op.name + 'Reply', op.reply, op.code, 'reply', None)
    h += cpp_event_variant(schema)
    h.append('}  // namespace webmetal::wire')
    h.append('')
    return '\n'.join(h) + '\n'


def cpp_struct(schema, s):
    out = []
    own = s.name in schema.to_guest and s.name not in schema.to_host
    doc = ('Wire form of `%s`. ' % s.metal if s.metal else '') + (s.doc or '')
    out += doc_lines(doc.strip(), '// ')
    out.append('struct %s {' % s.name)
    for f in s.fields:
        t = cpp_own_type(f.type) if own else cpp_ref_type(f.type)
        out.append('    %s %s%s;  // %s' % (t, f.cpp, cpp_default(f.type), field_doc(f)))
    out.append('};')
    out.append('')
    if s.name in schema.to_host:
        out.append('inline bool encodeFields(Writer& w, const %s& v) {' % s.name)
        for f in s.fields:
            out += cpp_encode(f.type, 'v.' + f.cpp, f.name, 4)
        out.append('    return true;')
        out.append('}')
        out.append('')
    if s.name in schema.to_guest:
        out.append('inline bool decodeFields(Reader& r, %s& v) {' % s.name)
        for f in s.fields:
            out += cpp_decode(f.type, 'v.' + f.cpp, f.name, 4)
        if s.name == 'Failure':
            out.append('    if (!failureValid(v.status, v.domain.has_value(), v.code, v.message.size())) return r.fail(DecodeError::BadFailure);')
        out.append('    return true;')
        out.append('}')
        out.append('')
    return out


def cpp_g2h_op(op):
    out = []
    doc = '0x%04x %s record.' % (op.code, op.channel)
    if op.methods:
        doc += ' Metal methods: %s.' % ', '.join('-[%s %s]' % tuple(m.split(' ')) for m in op.methods)
    if op.receiver:
        doc += ' %s is the receiver.' % op.receiver
    if op.result:
        doc += ' %s is the handle the guest allocated for the new object.' % op.result
    if op.doc:
        doc += ' ' + op.doc
    out += doc_lines(doc, '// ')
    out.append('struct %s {' % op.name)
    out.append('    static constexpr uint16_t kOpcode = 0x%04x;' % op.code)
    for f in op.fields:
        out.append('    %s %s%s;  // %s' % (cpp_ref_type(f.type), f.cpp, cpp_default(f.type), field_doc(f)))
    out.append('};')
    out.append('')
    if op.fields:
        out.append('inline bool encodeFields(Writer& w, const %s& v) {' % op.name)
    else:
        out.append('inline bool encodeFields(Writer&, const %s&) {' % op.name)
    for f in op.fields:
        if f.bound:
            out.append('    if (static_cast<uint64_t>(%s) != %s) return false;' % (cpp_len(f.type, 'v.' + f.cpp), cpp_bound_expr(f.bound, 'v.')))
    for f in op.fields:
        out += cpp_encode(f.type, 'v.' + f.cpp, f.name, 4)
    out.append('    return true;')
    out.append('}')
    out.append('')
    out.append('// Append the record; returns false (writer unchanged) if the value breaks a format rule.')
    out.append('inline bool encode(Writer& w, const %s& v) {' % op.name)
    out.append('    size_t start = w.begin(%s::kOpcode);' % op.name)
    out.append('    if (!encodeFields(w, v)) {')
    out.append('        w.abandon(start);')
    out.append('        return false;')
    out.append('    }')
    out.append('    return w.finish(start);')
    out.append('}')
    out.append('')
    return out


def cpp_h2g_record(name, fields, code, kind, doc):
    out = []
    out += doc_lines(('0x%04x %s. ' % (code, 'reply fields' if kind == 'reply' else 'event record')) + (doc or ''), '// ')
    out.append('struct %s {' % name)
    out.append('    static constexpr uint16_t kOpcode = 0x%04x;' % (code | 0x8000 if kind == 'reply' else code))
    for f in fields:
        out.append('    %s %s%s;  // %s' % (cpp_own_type(f.type), f.cpp, cpp_default(f.type), field_doc(f)))
    out.append('};')
    out.append('')
    if fields:
        out.append('inline bool decodeFields(Reader& r, %s& v) {' % name)
    else:
        out.append('inline bool decodeFields(Reader&, %s&) {' % name)
    for f in fields:
        out += cpp_decode(f.type, 'v.' + f.cpp, f.name, 4)
    out.append('    return true;')
    out.append('}')
    out.append('')
    if kind == 'reply':
        out.append('// Decode the reply to a %s request: either its fields or the failure.' % name[:-5])
        out.append('inline bool decodeReply(const uint8_t* data, size_t size, Reply<%s>& out, DecodeError* error = nullptr) {' % name)
        out.append('    return decodeReplyRecord(data, size, %s::kOpcode, out, error);' % name)
        out.append('}')
        out.append('')
    return out


def cpp_event_variant(schema):
    events = schema.by_channel['event']
    out = []
    out.append('// One decoded event record.')
    out.append('using Event = std::variant<%s>;' % ', '.join(op.name for op in events))
    out.append('')
    out.append('// Decode the event record at the start of data; stores its length in *consumed.')
    out.append('inline bool decodeEvent(const uint8_t* data, size_t size, Event& out, size_t* consumed, DecodeError* error = nullptr) {')
    out.append('    uint16_t opcode = 0;')
    out.append('    size_t length = 0;')
    out.append('    DecodeError local;')
    out.append('    DecodeError* e = error ? error : &local;')
    out.append('    if (!splitRecord(data, size, opcode, length, e)) return false;')
    out.append('    Reader r(data, length, e);')
    out.append('    switch (opcode) {')
    for op in events:
        out.append('    case %s::kOpcode: {' % op.name)
        out.append('        %s v;' % op.name)
        out.append('        if (!decodeFields(r, v) || !r.finish()) return false;')
        out.append('        out = std::move(v);')
        out.append('        break;')
        out.append('    }')
    out.append('    default:')
    out.append('        e->kind = DecodeError::UnknownOpcode;')
    out.append('        e->offset = 0;')
    out.append('        return false;')
    out.append('    }')
    out.append('    *consumed = length;')
    out.append('    return true;')
    out.append('}')
    out.append('')
    return out


CPP_RUNTIME = r'''#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#ifdef __OBJC__
#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#endif

namespace webmetal::wire {

constexpr size_t kHeaderLength = 8;
constexpr size_t kRecordAlignment = 8;
constexpr uint32_t kNilCount = 0xffffffffu;
constexpr uint16_t kReplyBit = 0x8000;
constexpr size_t kValueDepthLimit = 32;

// A non-owning array view.
template <class T>
struct Span {
    const T* data = nullptr;
    size_t size = 0;
};

// A non-owning byte string view.
struct Bytes {
    const uint8_t* data = nullptr;
    size_t size = 0;
};

// A non-owning string view; present = false encodes nil. For str fields the bytes must be UTF-8; for cstr fields they
// must not contain NUL.
struct Str {
    const char* data = nullptr;
    size_t size = 0;
    bool present = false;
};

inline Str str(const char* s) { return s ? Str{s, std::strlen(s), true} : Str{}; }
inline Str str(const std::string& s) { return Str{s.data(), s.size(), true}; }

template <class T>
struct Opt {
    bool present = false;
    T value{};
};

// A property-list value (see wire/src/value.rs). Dictionary entries are keys[i] -> values[i]; the encoder sorts them.
struct Value {
    enum class Kind : uint8_t { Bool = 1, Int = 2, UInt = 3, Float = 4, Double = 5, String = 6, Data = 7, Array = 8, Dictionary = 9 };
    Kind kind = Kind::Bool;
    bool boolean = false;
    int64_t i = 0;
    uint64_t u = 0;
    float f = 0;
    double d = 0;
    std::string string;
    std::vector<uint8_t> data;
    std::vector<Value> values;
    std::vector<std::string> keys;
};

inline bool validUtf8(const uint8_t* s, size_t n) {
    size_t i = 0;
    while (i < n) {
        uint8_t c = s[i];
        if (c < 0x80) {
            ++i;
            continue;
        }
        size_t len;
        uint32_t cp, min;
        if ((c & 0xe0) == 0xc0) { len = 2; cp = c & 0x1f; min = 0x80; }
        else if ((c & 0xf0) == 0xe0) { len = 3; cp = c & 0x0f; min = 0x800; }
        else if ((c & 0xf8) == 0xf0) { len = 4; cp = c & 0x07; min = 0x10000; }
        else return false;
        if (i + len > n) return false;
        for (size_t k = 1; k < len; ++k) {
            if ((s[i + k] & 0xc0) != 0x80) return false;
            cp = (cp << 6) | (s[i + k] & 0x3f);
        }
        if (cp < min || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
        i += len;
    }
    return true;
}

// The rules of a Failure (see Failure in the schema): status 0 has no domain, code 0 and an empty message; status 1
// (MetalError) has a domain; statuses 2, 3 and 4 have no domain and code 0.
inline bool failureValid(uint32_t status, bool hasDomain, int64_t code, size_t messageSize) {
    switch (status) {
    case 0: return !hasDomain && code == 0 && messageSize == 0;
    case 1: return hasDomain;
    case 2:
    case 3:
    case 4: return !hasDomain && code == 0;
    default: return false;
    }
}

// Appends records to a byte vector.
class Writer {
public:
    // The records written so far.
    std::vector<uint8_t> out;

    size_t begin(uint16_t opcode) {
        size_t start = out.size();
        u16raw(opcode);
        u16raw(0);
        u32raw(0);
        start_ = start;
        return start;
    }
    void abandon(size_t start) { out.resize(start); }
    bool finish(size_t start) {
        align(kRecordAlignment);
        size_t len = out.size() - start;
        if (len > (0xffffffffu & ~uint32_t(kRecordAlignment - 1))) {
            out.resize(start);
            return false;
        }
        uint32_t l = static_cast<uint32_t>(len);
        for (int k = 0; k < 4; ++k) out[start + 4 + k] = static_cast<uint8_t>(l >> (8 * k));
        return true;
    }

    void u8(uint8_t v) { out.push_back(v); }
    void boolean(bool v) { out.push_back(v ? 1 : 0); }
    void u16(uint16_t v) { align(2); u16raw(v); }
    void u32(uint32_t v) { align(4); u32raw(v); }
    void i32(int32_t v) { u32(static_cast<uint32_t>(v)); }
    void f32(float v) { uint32_t b; std::memcpy(&b, &v, 4); u32(b); }
    void u64(uint64_t v) { align(8); for (int k = 0; k < 8; ++k) out.push_back(static_cast<uint8_t>(v >> (8 * k))); }
    void i64(int64_t v) { u64(static_cast<uint64_t>(v)); }
    void f64(double v) { uint64_t b; std::memcpy(&b, &v, 8); u64(b); }
    void raw(const uint8_t* p, size_t n) { if (n) out.insert(out.end(), p, p + n); }

    bool count(size_t n) {
        if (n >= kNilCount) return false;
        u32(static_cast<uint32_t>(n));
        return true;
    }
    bool str(const Str& s) {
        if (!s.present || (s.size && !s.data)) return false;
        if (!validUtf8(reinterpret_cast<const uint8_t*>(s.data), s.size)) return false;
        if (!count(s.size)) return false;
        raw(reinterpret_cast<const uint8_t*>(s.data), s.size);
        return true;
    }
    bool optStr(const Str& s) {
        if (!s.present) { u32(kNilCount); return true; }
        return str(s);
    }
    bool optCstr(const Str& s) {
        if (!s.present) { u32(kNilCount); return true; }
        if (s.size && !s.data) return false;
        if (s.size && std::memchr(s.data, 0, s.size)) return false;
        if (!count(s.size)) return false;
        raw(reinterpret_cast<const uint8_t*>(s.data), s.size);
        return true;
    }
    bool bytes(const Bytes& b) {
        if (b.size && !b.data) return false;
        if (!count(b.size)) return false;
        raw(b.data, b.size);
        return true;
    }
    bool optValue(const Value* v) {
        if (!v) { u8(0); return true; }
        return value(*v, 1);
    }

private:
    size_t start_ = 0;
    void align(size_t a) { while ((out.size() - start_) % a) out.push_back(0); }
    void u16raw(uint16_t v) { out.push_back(static_cast<uint8_t>(v)); out.push_back(static_cast<uint8_t>(v >> 8)); }
    void u32raw(uint32_t v) { for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(v >> (8 * k))); }
    bool value(const Value& v, size_t depth) {
        if (depth > kValueDepthLimit) return false;
        u8(static_cast<uint8_t>(v.kind));
        switch (v.kind) {
        case Value::Kind::Bool: boolean(v.boolean); return true;
        case Value::Kind::Int: i64(v.i); return true;
        case Value::Kind::UInt: u64(v.u); return true;
        case Value::Kind::Float: f32(v.f); return true;
        case Value::Kind::Double: f64(v.d); return true;
        case Value::Kind::String: return str(Str{v.string.data(), v.string.size(), true});
        case Value::Kind::Data: return bytes(Bytes{v.data.data(), v.data.size()});
        case Value::Kind::Array:
            if (!count(v.values.size())) return false;
            for (const Value& e : v.values) if (!value(e, depth + 1)) return false;
            return true;
        case Value::Kind::Dictionary: {
            if (v.keys.size() != v.values.size()) return false;
            std::vector<size_t> order(v.keys.size());
            for (size_t k = 0; k < order.size(); ++k) order[k] = k;
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return v.keys[a] < v.keys[b]; });
            for (size_t k = 1; k < order.size(); ++k)
                if (v.keys[order[k - 1]] == v.keys[order[k]]) return false;
            if (!count(order.size())) return false;
            for (size_t k : order) {
                if (!str(Str{v.keys[k].data(), v.keys[k].size(), true})) return false;
                if (!value(v.values[k], depth + 1)) return false;
            }
            return true;
        }
        }
        return false;
    }
};

struct DecodeError {
    enum Kind { None, Truncated, BadLength, BadFlags, UnknownOpcode, WrongChannel, NonZeroPadding, BadBool, NullHandle,
                BadString, BadCount, BadFailure };
    Kind kind = None;
    size_t offset = 0;
};

// Reads the fields of one record (header included in data).
class Reader {
public:
    Reader(const uint8_t* data, size_t size, DecodeError* error) : data_(data), size_(size), at_(kHeaderLength), error_(error) {}

    bool fail(DecodeError::Kind kind) {
        error_->kind = kind;
        error_->offset = at_;
        return false;
    }
    bool u8(uint8_t& v) { if (at_ + 1 > size_) return fail(DecodeError::Truncated); v = data_[at_++]; return true; }
    bool boolean(bool& v) {
        uint8_t b;
        if (!u8(b)) return false;
        if (b > 1) { --at_; return fail(DecodeError::BadBool); }
        v = b == 1;
        return true;
    }
    bool u16(uint16_t& v) { if (!align(2) || !need(2)) return false; v = uint16_t(data_[at_] | data_[at_ + 1] << 8); at_ += 2; return true; }
    bool u32(uint32_t& v) {
        if (!align(4) || !need(4)) return false;
        v = 0;
        for (int k = 0; k < 4; ++k) v |= uint32_t(data_[at_ + k]) << (8 * k);
        at_ += 4;
        return true;
    }
    bool i32(int32_t& v) { uint32_t u; if (!u32(u)) return false; v = static_cast<int32_t>(u); return true; }
    bool f32(float& v) { uint32_t u; if (!u32(u)) return false; std::memcpy(&v, &u, 4); return true; }
    bool u64(uint64_t& v) {
        if (!align(8) || !need(8)) return false;
        v = 0;
        for (int k = 0; k < 8; ++k) v |= uint64_t(data_[at_ + k]) << (8 * k);
        at_ += 8;
        return true;
    }
    bool i64(int64_t& v) { uint64_t u; if (!u64(u)) return false; v = static_cast<int64_t>(u); return true; }
    bool f64(double& v) { uint64_t u; if (!u64(u)) return false; std::memcpy(&v, &u, 8); return true; }
    bool raw(uint8_t* out, size_t n) { if (!need(n)) return false; std::memcpy(out, data_ + at_, n); at_ += n; return true; }
    bool count(uint32_t& n) {
        if (!u32(n)) return false;
        if (n == kNilCount || n > size_ - at_) { at_ -= 4; return fail(DecodeError::BadCount); }
        return true;
    }
    bool str(std::string& s) {
        uint32_t n;
        if (!count(n)) return false;
        if (!validUtf8(data_ + at_, n)) return fail(DecodeError::BadString);
        s.assign(reinterpret_cast<const char*>(data_ + at_), n);
        at_ += n;
        return true;
    }
    bool optStr(std::optional<std::string>& s) {
        size_t save = at_;
        uint32_t n;
        if (!u32(n)) return false;
        if (n == kNilCount) { s.reset(); return true; }
        at_ = save;
        s.emplace();
        return str(*s);
    }
    bool bytes(std::vector<uint8_t>& b) {
        uint32_t n;
        if (!count(n)) return false;
        b.assign(data_ + at_, data_ + at_ + n);
        at_ += n;
        return true;
    }
    bool finish() {
        if (!align(kRecordAlignment)) return false;
        if (at_ != size_) return fail(DecodeError::BadLength);
        return true;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t at_;
    DecodeError* error_;
    bool need(size_t n) { return at_ + n <= size_ ? true : fail(DecodeError::Truncated); }
    bool align(size_t a) {
        while (at_ % a) {
            if (at_ >= size_) return fail(DecodeError::Truncated);
            if (data_[at_] != 0) return fail(DecodeError::NonZeroPadding);
            ++at_;
        }
        return true;
    }
};

// Check the header of the record at the start of data.
inline bool splitRecord(const uint8_t* data, size_t size, uint16_t& opcode, size_t& length, DecodeError* error) {
    if (size < kHeaderLength) { error->kind = DecodeError::Truncated; error->offset = 0; return false; }
    opcode = uint16_t(data[0] | data[1] << 8);
    uint16_t flags = uint16_t(data[2] | data[3] << 8);
    uint32_t len = uint32_t(data[4]) | uint32_t(data[5]) << 8 | uint32_t(data[6]) << 16 | uint32_t(data[7]) << 24;
    if (flags) { error->kind = DecodeError::BadFlags; error->offset = 2; return false; }
    if (len < kHeaderLength || len % kRecordAlignment || len > size) { error->kind = DecodeError::BadLength; error->offset = 4; return false; }
    length = len;
    return true;
}

struct Failure;

// The outcome of a control request: ok and its reply fields, or the failure.
template <class T>
struct Reply;

'''


CPP_RUNTIME_TAIL = r'''
#ifdef __OBJC__
// A string field from an NSString: nil maps to nil. Returns false when the string has no UTF-8 form (unpaired
// surrogates). The bytes live as long as the autorelease pool that owns -UTF8String's buffer.
inline bool strFromNSString(NSString* s, Str& out) {
    if (!s) {
        out = Str{};
        return true;
    }
    const char* utf8 = [s UTF8String];
    if (!utf8) return false;
    out = Str{utf8, std::strlen(utf8), true};
    return true;
}

// A property-list value from a Foundation object graph of NSDictionary (NSString keys), NSArray, NSString, NSData and
// NSNumber. Returns false for any other class, a non-string key, a string without a UTF-8 form, or nesting deeper than
// kValueDepthLimit; the caller reports that, it never drops the value.
inline bool valueFromObject(id object, Value& out, size_t depth = 1) {
    if (!object || depth > kValueDepthLimit) return false;
    if ([object isKindOfClass:[NSNumber class]]) {
        NSNumber* n = object;
        if (CFGetTypeID((__bridge CFTypeRef)n) == CFBooleanGetTypeID()) {
            out.kind = Value::Kind::Bool;
            out.boolean = n.boolValue;
            return true;
        }
        switch (n.objCType[0]) {
        case 'c': case 's': case 'i': case 'l': case 'q':
            out.kind = Value::Kind::Int;
            out.i = n.longLongValue;
            return true;
        case 'C': case 'S': case 'I': case 'L': case 'Q':
            out.kind = Value::Kind::UInt;
            out.u = n.unsignedLongLongValue;
            return true;
        case 'f':
            out.kind = Value::Kind::Float;
            out.f = n.floatValue;
            return true;
        case 'd':
            out.kind = Value::Kind::Double;
            out.d = n.doubleValue;
            return true;
        default:
            return false;
        }
    }
    if ([object isKindOfClass:[NSString class]]) {
        const char* utf8 = [(NSString*)object UTF8String];
        if (!utf8) return false;
        out.kind = Value::Kind::String;
        out.string = utf8;
        return true;
    }
    if ([object isKindOfClass:[NSData class]]) {
        NSData* d = object;
        out.kind = Value::Kind::Data;
        out.data.assign(static_cast<const uint8_t*>(d.bytes), static_cast<const uint8_t*>(d.bytes) + d.length);
        return true;
    }
    if ([object isKindOfClass:[NSArray class]]) {
        out.kind = Value::Kind::Array;
        for (id item in (NSArray*)object) {
            Value v;
            if (!valueFromObject(item, v, depth + 1)) return false;
            out.values.push_back(std::move(v));
        }
        return true;
    }
    if ([object isKindOfClass:[NSDictionary class]]) {
        NSDictionary* dict = object;
        out.kind = Value::Kind::Dictionary;
        for (id key in dict) {
            if (![key isKindOfClass:[NSString class]]) return false;
            const char* utf8 = [(NSString*)key UTF8String];
            if (!utf8) return false;
            Value v;
            if (!valueFromObject(dict[key], v, depth + 1)) return false;
            out.keys.push_back(utf8);
            out.values.push_back(std::move(v));
        }
        return true;
    }
    return false;
}
#endif

template <class T>
struct Reply {
    bool ok = false;
    T value{};
    Failure failure{};
};

template <class T>
inline bool decodeReplyRecord(const uint8_t* data, size_t size, uint16_t expectedOpcode, Reply<T>& out, DecodeError* error) {
    DecodeError local;
    DecodeError* e = error ? error : &local;
    uint16_t opcode = 0;
    size_t length = 0;
    if (!splitRecord(data, size, opcode, length, e)) return false;
    if (opcode != expectedOpcode || length != size) { e->kind = DecodeError::WrongChannel; e->offset = 0; return false; }
    Reader r(data, length, e);
    uint32_t status = 0;
    if (!r.u32(status)) return false;
    if (status == 0) {
        out.ok = true;
        out.failure = Failure{};
        return decodeFields(r, out.value) && r.finish();
    }
    out.ok = false;
    Reader again(data, length, e);
    return decodeFields(again, out.failure) && again.finish();
}
'''


def gen_cpp_full(schema):
    text = gen_cpp(schema)
    # Reply<T> needs the complete Failure type; place its definition right after Failure's codec.
    marker = 'inline bool decodeFields(Reader& r, Failure& v) {'
    at = text.index(marker)
    end = text.index('}\n\n', at) + 3
    text = text[:end] + CPP_RUNTIME_TAIL + '\n' + text[end:]
    return text


# C++ fuzz --------------------------------------------------------------------------------------------------------------

def cpp_random_ref(t, x, ind, length=None):
    """Statements filling guest-to-host place x with a random value (storage lives in the arena)."""
    k = t.kind
    p = ' ' * ind
    if k == 'scalar':
        return [p + '%s = g.%s();' % (x, 'boolean' if t.name == 'bool' else t.name)]
    if k == 'flag':
        return [p + '%s = g.boolean();' % x]
    if k == 'guest_addr':
        return [p + '%s = g.u64();' % x]
    if k == 'surface':
        return [p + '%s = g.u32();' % x]
    if k == 'handle':
        return [p + '%s = g.%s();' % (x, 'optHandle' if t.nullable else 'handle')]
    if k == 'struct':
        return [p + 'random(g, %s);' % x]
    if k == 'opt_struct':
        return [p + '%s.present = g.boolean();' % x, p + 'if (%s.present) random(g, %s.value);' % (x, x)]
    if k == 'fixed':
        return [p + 'for (auto& b : %s) b = g.u8();' % x]
    if k == 'str':
        return [p + '%s = g.string(%s);' % (x, length or 'g.stringLength()')]
    if k == 'opt_str':
        return [p + '%s = %s;' % (x, 'g.optString()' if length is None else 'g.string(%s)' % length)]
    if k == 'opt_cstr':
        return [p + '%s = %s;' % (x, 'g.optCstr()' if length is None else 'g.cstr(%s)' % length)]
    if k == 'bytes':
        return [p + '%s = g.bytes(%s);' % (x, length or 'g.bytesLength()')]
    if k == 'opt_value':
        return [p + '%s = g.optValue();' % x]
    if k == 'array':
        et = cpp_ref_type(t.elem)
        return ([p + '{', p + '    uint64_t n = %s;' % (length or 'g.arrayLength()'),
                 p + '    std::vector<%s>& items = g.arena<%s>(n);' % (et, et), p + '    for (auto& item : items) {']
                + cpp_random_ref(t.elem, 'item', ind + 8) + [p + '    }', p + '    %s = Span<%s>{items.data(), items.size()};' % (x, et), p + '}'])
    raise AssertionError(k)


def cpp_random_own(t, x, ind):
    k = t.kind
    p = ' ' * ind
    if k in ('scalar', 'flag', 'guest_addr', 'surface', 'handle', 'struct', 'fixed'):
        return cpp_random_ref(t, x, ind)
    if k == 'opt_struct':
        return [p + 'if (g.boolean()) {', p + '    %s.emplace();' % x, p + '    random(g, *%s);' % x, p + '} else {',
                p + '    %s.reset();' % x, p + '}']
    if k == 'str':
        return [p + '%s = g.ownString(g.stringLength());' % x]
    if k == 'opt_str':
        return [p + '%s = g.ownOptString();' % x]
    if k == 'bytes':
        return [p + '%s = g.ownBytes(g.bytesLength());' % x]
    if k == 'array':
        return ([p + '{', p + '    uint64_t n = g.arrayLength();', p + '    %s.assign(n, {});' % x, p + '    for (auto& item : %s) {' % x]
                + cpp_random_own(t.elem, 'item', ind + 8) + [p + '    }', p + '}'])
    raise AssertionError(k)


def cpp_same(t, a, b):
    k = t.kind
    if k == 'scalar' and t.name in ('f32', 'f64'):
        return 'sameBits(%s, %s)' % (a, b)
    if k in ('scalar', 'flag', 'guest_addr', 'surface', 'handle', 'fixed', 'str', 'opt_str', 'bytes'):
        return '%s == %s' % (a, b)
    if k == 'struct':
        return 'same(%s, %s)' % (a, b)
    if k == 'opt_struct':
        return '(%s.has_value() == %s.has_value() && (!%s.has_value() || same(*%s, *%s)))' % (a, b, a, a, b)
    if k == 'array':
        return 'sameVector(%s, %s, [](const auto& x, const auto& y) { return %s; })' % (a, b, cpp_same(t.elem, 'x', 'y'))
    raise AssertionError(k)


def gen_cpp_fuzz(schema):
    h = []
    h.append('// Generated by wire/gen/wiregen.py from wire/schema/webmetal.toml. Do not edit.')
    h.append('//')
    h.append('// Random record generators and comparisons for the cross-language tests. Draws come from splitmix64 in the same')
    h.append('// order as the Rust generators (wire/src/generated.rs), so one seed yields the same records in both languages.')
    h.append('')
    h.append('#pragma once')
    h.append('')
    h.append('#include "WebMetalWire.h"')
    h.append('')
    h.append(CPP_FUZZ_RUNTIME)
    for s in schema.struct_order:
        if s.name in schema.to_host:
            h.append('inline void random(Gen& g, %s& v) {' % s.name)
            for f in s.fields:
                h += cpp_random_ref(f.type, 'v.' + f.cpp, 4)
            h.append('}')
            h.append('')
        if s.name in schema.to_guest:
            if s.name == 'Failure':
                h.append('inline void random(Gen& g, Failure& v) { randomFailure(g, v); }')
                h.append('')
            elif s.name not in schema.to_host:
                h.append('inline void random(Gen& g, %s& v) {' % s.name)
                for f in s.fields:
                    h += cpp_random_own(f.type, 'v.' + f.cpp, 4)
                h.append('}')
                h.append('')
            h.append('inline bool same(const %s& a, const %s& b) {' % (s.name, s.name))
            h.append('    return %s;' % (' &&\n           '.join(cpp_same(f.type, 'a.' + f.cpp, 'b.' + f.cpp) for f in s.fields) or 'true'))
            h.append('}')
            h.append('')
    for op in schema.ops:
        if op.channel in ('control', 'stream'):
            groups = bound_groups(op.fields)
            h.append(('inline void random(Gen& g, %s& v) {' if op.fields else 'inline void random(Gen&, %s&) {') % op.name)
            for f in op.fields:
                if not groups or not f.type.variable:
                    h += cpp_random_ref(f.type, 'v.' + f.cpp, 4)
            for i, (bound, bytes_like) in enumerate(groups):
                h.append('    uint64_t len%d = g.%s();' % (i, 'bytesLength' if bytes_like else 'arrayLength'))
                h.append('    %s = len%d;' % (cpp_bound_expr(bound, 'v.'), i))
            for f in op.fields:
                if groups and f.type.variable:
                    length = 'len%d' % [g[0] for g in groups].index(f.bound) if f.bound else None
                    h += cpp_random_ref(f.type, 'v.' + f.cpp, 4, length)
            h.append('}')
            h.append('')
        if op.channel == 'event' or op.channel == 'control':
            name, fields = (op.name, op.fields) if op.channel == 'event' else (op.name + 'Reply', op.reply)
            h.append(('inline void random(Gen& g, %s& v) {' if fields else 'inline void random(Gen&, %s&) {') % name)
            for f in fields:
                h += cpp_random_own(f.type, 'v.' + f.cpp, 4)
            h.append('}')
            h.append('')
            h.append(('inline bool same(const %s& a, const %s& b) {' if fields else 'inline bool same(const %s&, const %s&) {') % (name, name))
            h.append('    return %s;' % (' &&\n           '.join(cpp_same(f.type, 'a.' + f.cpp, 'b.' + f.cpp) for f in fields) or 'true'))
            h.append('}')
            h.append('')
    # channel-level random encoders and checkers
    for channel in ('control', 'stream'):
        ops = schema.by_channel[channel]
        h.append('// Append one random %s record, the op chosen uniformly in schema order.' % channel)
        h.append('inline bool encodeRandom%s(Gen& g, Writer& w) {' % ('Request' if channel == 'control' else 'Command'))
        h.append('    switch (g.below(%d)) {' % len(ops))
        for i, op in enumerate(ops):
            h.append('    case %d: { %s v; random(g, v); return encode(w, v); }' % (i, op.name))
        h.append('    }')
        h.append('    return false;')
        h.append('}')
        h.append('')
    ctrl = schema.by_channel['control']
    h.append('// Decode the next reply record and compare it with the next random reply; false on any difference.')
    h.append('inline bool checkRandomReply(Gen& g, const uint8_t* data, size_t size, size_t* consumed) {')
    h.append('    uint16_t opcode = 0;')
    h.append('    size_t length = 0;')
    h.append('    DecodeError e;')
    h.append('    if (!splitRecord(data, size, opcode, length, &e)) return false;')
    h.append('    *consumed = length;')
    h.append('    switch (g.below(%d)) {' % len(ctrl))
    for i, op in enumerate(ctrl):
        h.append('    case %d: {' % i)
        h.append('        Reply<%sReply> expected;' % op.name)
        h.append('        if (g.below(4) == 0) { randomErrorFailure(g, expected.failure); } else { expected.ok = true; random(g, expected.value); }')
        h.append('        Reply<%sReply> got;' % op.name)
        h.append('        if (!decodeReply(data, length, got, &e)) return false;')
        h.append('        return got.ok == expected.ok && (got.ok ? same(got.value, expected.value) : same(got.failure, expected.failure));')
        h.append('    }')
    h.append('    }')
    h.append('    return false;')
    h.append('}')
    h.append('')
    events = schema.by_channel['event']
    h.append('// Decode the next event record and compare it with the next random event; false on any difference.')
    h.append('inline bool checkRandomEvent(Gen& g, const uint8_t* data, size_t size, size_t* consumed) {')
    h.append('    Event got;')
    h.append('    if (!decodeEvent(data, size, got, consumed)) return false;')
    h.append('    switch (g.below(%d)) {' % len(events))
    for i, op in enumerate(events):
        h.append('    case %d: {' % i)
        h.append('        %s expected;' % op.name)
        h.append('        random(g, expected);')
        h.append('        const %s* v = std::get_if<%s>(&got);' % (op.name, op.name))
        h.append('        return v && same(*v, expected);')
        h.append('    }')
    h.append('    }')
    h.append('    return false;')
    h.append('}')
    h.append('')
    h.append('}  // namespace webmetal::wire::fuzz')
    h.append('')
    return '\n'.join(h) + '\n'


CPP_FUZZ_RUNTIME = r'''#include <deque>
#include <memory>

namespace webmetal::wire::fuzz {

// splitmix64 (Steele, Lea and Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014), and the value
// draws shared with wire/src/fuzz.rs.
class Gen {
public:
    explicit Gen(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        state_ += 0x9e3779b97f4a7c15ull;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t n) { return next() % n; }
    uint8_t u8() { return static_cast<uint8_t>(next()); }
    uint16_t u16() { return static_cast<uint16_t>(next()); }
    uint32_t u32() { return static_cast<uint32_t>(next()); }
    uint64_t u64() { return next(); }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    int64_t i64() { return static_cast<int64_t>(next()); }
    float f32() { uint32_t b = u32(); float f; std::memcpy(&f, &b, 4); return f; }
    double f64() { uint64_t b = next(); double d; std::memcpy(&d, &b, 8); return d; }
    bool boolean() { return (next() & 1) == 1; }
    uint64_t handle() { uint64_t v = next(); return v ? v : 1; }
    uint64_t optHandle() { return below(4) == 0 ? 0 : handle(); }
    uint64_t stringLength() { return below(17); }
    uint64_t bytesLength() { return below(33); }
    uint64_t arrayLength() { return below(5); }

    std::string ownString(uint64_t codePoints) {
        std::string s;
        for (uint64_t k = 0; k < codePoints; ++k) appendUtf8(s, codePoint());
        return s;
    }
    std::optional<std::string> ownOptString() {
        if (below(4) == 0) return std::nullopt;
        return ownString(stringLength());
    }
    std::vector<uint8_t> ownBytes(uint64_t n) {
        std::vector<uint8_t> b(n);
        for (auto& x : b) x = u8();
        return b;
    }
    Str string(uint64_t codePoints) {
        strings_.push_back(ownString(codePoints));
        return Str{strings_.back().data(), strings_.back().size(), true};
    }
    Str optString() { return below(4) == 0 ? Str{} : string(stringLength()); }
    Str cstr(uint64_t n) {
        std::string s;
        for (uint64_t k = 0; k < n; ++k) s.push_back(static_cast<char>(1 + below(255)));
        strings_.push_back(std::move(s));
        return Str{strings_.back().data(), strings_.back().size(), true};
    }
    Str optCstr() { return below(4) == 0 ? Str{} : cstr(stringLength()); }
    Bytes bytes(uint64_t n) {
        blobs_.push_back(ownBytes(n));
        return Bytes{blobs_.back().data(), blobs_.back().size()};
    }
    const Value* optValue() {
        if (below(4) == 0) return nullptr;
        values_.push_back(value(1));
        return &values_.back();
    }
    template <class T>
    std::vector<T>& arena(uint64_t n) {
        auto holder = std::make_shared<std::vector<T>>(n);
        arenas_.push_back(holder);
        return *holder;
    }

private:
    uint64_t state_;
    std::deque<std::string> strings_;
    std::deque<std::vector<uint8_t>> blobs_;
    std::deque<Value> values_;
    std::deque<std::shared_ptr<void>> arenas_;

    uint32_t codePoint() {
        switch (below(4)) {
        case 0: return 0x20 + static_cast<uint32_t>(below(0x5f));
        case 1: return 0x80 + static_cast<uint32_t>(below(0x780));
        case 2: {
            uint32_t cp = 0x800 + static_cast<uint32_t>(below(0xf000));
            return cp >= 0xd800 ? cp + 0x800 : cp;
        }
        default: return 0x10000 + static_cast<uint32_t>(below(0x100000));
        }
    }
    static void appendUtf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            s.push_back(static_cast<char>(0xf0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }
    Value value(size_t depth) {
        Value v;
        switch (below(depth < 3 ? 9 : 7)) {
        case 0: v.kind = Value::Kind::Bool; v.boolean = boolean(); break;
        case 1: v.kind = Value::Kind::Int; v.i = i64(); break;
        case 2: v.kind = Value::Kind::UInt; v.u = u64(); break;
        case 3: v.kind = Value::Kind::Float; v.f = f32(); break;
        case 4: v.kind = Value::Kind::Double; v.d = f64(); break;
        case 5: v.kind = Value::Kind::String; v.string = ownString(stringLength()); break;
        case 6: v.kind = Value::Kind::Data; v.data = ownBytes(bytesLength()); break;
        case 7: {
            v.kind = Value::Kind::Array;
            uint64_t n = below(4);
            for (uint64_t k = 0; k < n; ++k) v.values.push_back(value(depth + 1));
            break;
        }
        default: {
            v.kind = Value::Kind::Dictionary;
            uint64_t n = below(4);
            std::vector<std::pair<std::string, Value>> entries;
            for (uint64_t k = 0; k < n; ++k) {
                std::string key = ownString(stringLength());
                Value item = value(depth + 1);
                entries.emplace_back(std::move(key), std::move(item));
            }
            std::stable_sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (auto& e : entries) {
                if (!v.keys.empty() && v.keys.back() == e.first) continue;
                v.keys.push_back(e.first);
                v.values.push_back(std::move(e.second));
            }
            break;
        }
        }
        return v;
    }
};

inline bool sameBits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }
inline bool sameBits(double a, double b) { return std::memcmp(&a, &b, 8) == 0; }

template <class T, class F>
inline bool sameVector(const std::vector<T>& a, const std::vector<T>& b, F eq) {
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k)
        if (!eq(a[k], b[k])) return false;
    return true;
}

// A random valid Failure: status drawn first, then the fields the status allows.
inline void randomFailureWithStatus(Gen& g, Failure& f, uint32_t status) {
    f.status = status;
    if (status == 0) {
        f.domain.reset();
        f.code = 0;
        f.message.clear();
        return;
    }
    if (status == 1) {
        f.domain = g.ownString(g.stringLength());
        f.code = g.i64();
    } else {
        f.domain.reset();
        f.code = 0;
    }
    f.message = g.ownString(g.stringLength());
}
inline void randomFailure(Gen& g, Failure& f) { randomFailureWithStatus(g, f, static_cast<uint32_t>(g.below(5))); }
inline void randomErrorFailure(Gen& g, Failure& f) { randomFailureWithStatus(g, f, 1 + static_cast<uint32_t>(g.below(4))); }

'''


# Documentation ---------------------------------------------------------------------------------------------------------

def gen_md(schema):
    o = ['# WebMetal wire opcodes', '',
         'Generated by `wire/gen/wiregen.py` from `wire/schema/webmetal.toml`; do not edit. Schema SHA-256 `%s`.' % schema.hash.hex(),
         '', 'Field types and the record format are described in `wire/README.md`.', '']
    titles = {'control': 'Control requests (guest to host, one reply each)', 'stream': 'Command-stream records',
              'event': 'Events (host to guest)'}
    for channel in ('control', 'stream', 'event'):
        o.append('## %s' % titles[channel])
        o.append('')
        o.append('| Opcode | Record | Metal methods | Fields | Reply |' if channel == 'control' else '| Opcode | Record | Metal methods | Fields |')
        o.append('| --- | --- | --- | --- | --- |' if channel == 'control' else '| --- | --- | --- | --- |')
        for op in schema.by_channel[channel]:
            methods = '<br>'.join('`-[%s %s]`' % tuple(m.split(' ')) for m in op.methods) or '(plugin)'
            fields = '<br>'.join('`%s: %s`%s' % (f.name, f.type_text.replace('|', '\\|'), ' = `%s`' % f.bound if f.bound else '')
                                 for f in op.fields) or '(none)'
            row = '| `0x%04x` | %s | %s | %s |' % (op.code, op.name, methods, fields)
            if channel == 'control':
                row += ' %s |' % ('<br>'.join('`%s: %s`' % (f.name, f.type_text) for f in op.reply) or '(status only)')
            o.append(row)
        o.append('')
    return '\n'.join(o)


def outputs():
    schema = Schema(SCHEMA)
    return {
        OUT_RS: gen_rust(schema),
        OUT_H: gen_cpp_full(schema),
        OUT_FUZZ_H: gen_cpp_fuzz(schema),
        OUT_MD: gen_md(schema),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='exit 1 if a committed output is stale')
    args = parser.parse_args()
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    for path in stale:
        print('stale: %s' % path.relative_to(WIRE.parent))
    return 1 if stale else 0


if __name__ == '__main__':
    sys.exit(main())
