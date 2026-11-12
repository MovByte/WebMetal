#!/usr/bin/env python3
"""Generate host/src/exec/schema.rs, the executor's object and scope rules, from wire/schema/webmetal.toml.

The schema is read with the wire generator's own parser (wire/gen/wiregen.py), so field names, types and op order are
the ones the decoder is built from. The output holds:

  Kind              the object kinds of the schema and which kinds a field of each kind accepts
  stream_rule       for every stream opcode, the encoder scope it belongs to and the encoder it opens
  request_handles   every object handle a control request carries, with its kind and role
  command_handles   the same for a command-stream record
  failure_reply     the reply record of any control opcode carrying a failure
  reply_failure     the failure a reply carries

Scopes come from the Metal protocol of the methods each stream op carries; an op whose methods span two scopes, or a
protocol this generator does not know, stops the generator, so a schema change cannot be classified silently.

Usage:
  execgen.py            write schema.rs
  execgen.py --check    exit 1 if schema.rs differs from what the schema generates
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE.parent / 'schema.rs'
sys.path.insert(0, str(ROOT / 'wire' / 'gen'))

from wiregen import SCHEMA, Schema, rust_ident  # noqa: E402

# Stream ops: the scope each Metal protocol's methods belong to.
PROTOCOL_SCOPE = {
    'MTLCommandBuffer': 'CommandBuffer',
    'MTLCommandBufferSPI': 'CommandBuffer',
    'MTLCommandEncoder': 'AnyEncoder',
    'MTLRenderCommandEncoder': 'Render',
    'MTLRenderCommandEncoderSPI': 'Render',
    'MTLComputeCommandEncoder': 'Compute',
    'MTLComputeCommandEncoderSPI': 'Compute',
    'MTLBlitCommandEncoder': 'Blit',
    'MTLBlitCommandEncoderSPI': 'Blit',
    'MTLParallelRenderCommandEncoder': 'Parallel',
}

# Encoder-creating stream ops: the selector that creates the encoder, and the encoder kind it opens.
OPENS = {
    'MTLCommandBuffer renderCommandEncoderWithDescriptor:': 'Render',
    'MTLCommandBuffer parallelRenderCommandEncoderWithDescriptor:': 'Parallel',
    'MTLParallelRenderCommandEncoder renderCommandEncoder': 'ParallelSub',
    'MTLCommandBuffer computeCommandEncoder': 'Compute',
    'MTLCommandBuffer computeCommandEncoderWithDispatchType:': 'Compute',
    'MTLCommandBuffer computeCommandEncoderWithDescriptor:': 'Compute',
    'MTLCommandBufferSPI computeCommandEncoderWithDispatchType:substreamCount:': 'Compute',
    'MTLCommandBuffer blitCommandEncoder': 'Blit',
    'MTLCommandBuffer blitCommandEncoderWithDescriptor:': 'Blit',
}

# Plugin requests that bind a new handle without naming a Metal method: their docs in the schema say so
# ("Name a shared event ... so stream records can refer to it by handle", "Name a function of a registered AIR module").
CREATES = {'RegisterSharedEvent': 'event', 'RegisterFunction': 'function'}
# The request whose handle is the object the guest deallocated.
RELEASES = {'Release': 'object'}


def scope_of(op):
    scopes = set()
    for m in op.methods:
        proto = m.split(' ')[0]
        if proto not in PROTOCOL_SCOPE:
            raise SystemExit('%s: method %r of an unknown protocol' % (op.name, m))
        scopes.add(PROTOCOL_SCOPE[proto])
    if len(scopes) != 1:
        raise SystemExit('%s: methods span scopes %s' % (op.name, sorted(scopes)))
    return scopes.pop()


def opens_of(op):
    if not op.result:
        return None
    kinds = {OPENS.get(m) for m in op.methods}
    if None in kinds or len(kinds) != 1:
        raise SystemExit('%s: creates %s through methods %s this generator does not classify'
                         % (op.name, op.result, op.methods))
    return kinds.pop()


def role_of(op, field):
    if op.result == field.name or CREATES.get(op.name) == field.name:
        return 'Result'
    if RELEASES.get(op.name) == field.name:
        return 'Released'
    if op.receiver == field.name:
        return 'Receiver'
    return 'Argument'


class Visitor:
    """Rust statements that report every handle reachable from a value."""

    def __init__(self, schema):
        self.schema = schema

    def has_handles(self, t, seen=()):
        if t.kind == 'handle':
            return True
        if t.kind == 'array':
            return self.has_handles(t.elem, seen)
        if t.kind in ('struct', 'opt_struct'):
            if t.name in seen:
                return False
            return any(self.has_handles(f.type, seen + (t.name,)) for f in self.schema.structs[t.name].fields)
        return False

    def visit(self, t, expr, path, role, ind, borrowed):
        """Statements reporting the handles of `expr`, a place of type t, or a reference to one when `borrowed`."""
        p = ' ' * ind
        deref = '*' + expr if borrowed else expr
        if t.kind == 'handle':
            use = 'HandleUse { field: "%s", kind: Kind::%s, role: Role::%s, handle: %%s }' % (path, t.name, role)
            if t.nullable:
                return [p + 'if let Some(h) = %s {' % deref, p + '    f(' + use % 'h' + ');', p + '}']
            return [p + 'f(' + use % deref + ');']
        if t.kind == 'array':
            out = [p + 'for e in %s {' % (expr if borrowed else '&' + expr)]
            out += self.visit(t.elem, 'e', path + '[]', role, ind + 4, True)
            out.append(p + '}')
            return out
        if t.kind in ('struct', 'opt_struct'):
            s = self.schema.structs[t.name]
            if t.kind == 'opt_struct':
                out = [p + 'if let Some(s) = %s {' % (expr if borrowed else '&' + expr)]
                base, inner = 's', ind + 4
            else:
                out, base, inner = [], expr, ind
            for f in s.fields:
                if self.has_handles(f.type):
                    out += self.visit(f.type, '%s.%s' % (base, f.rust), path + '.' + f.name, role, inner, False)
            if t.kind == 'opt_struct':
                out.append(p + '}')
            return out
        return []


def gen(schema):
    o = []
    w = o.append
    w('// Generated by host/src/exec/gen/execgen.py from wire/schema/webmetal.toml. Do not edit.')
    w('//')
    w('// Schema SHA-256: %s' % schema.hash.hex())
    w('')
    w('use webmetal_wire::*;')
    w('')
    w('/// SHA-256 of the schema this file was generated from; equal to [`webmetal_wire::SCHEMA_HASH`].')
    w('pub const SCHEMA_HASH: [u8; 32] = [%s];' % ', '.join('0x%02x' % b for b in schema.hash))
    w('')

    # Kinds ------------------------------------------------------------------------------------------------------
    objects = list(schema.objects.values())
    created = set()
    for op in schema.ops:
        for f in op.fields:
            if f.type.kind == 'handle' and role_of(op, f) == 'Result':
                created.add(f.type.name)
    # A command buffer becomes known to the host when its Commit names it (schema: "A command buffer is a guest object
    # until Commit").
    created.add('CommandBuffer')
    any_kind = [o['name'] for o in objects if not o['protocols']]
    if any_kind != ['Object']:
        raise SystemExit('expected exactly one kind without protocols (Object), found %s' % any_kind)
    w('/// An object kind of the schema (`[[object]]`).')
    w('#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]')
    w('pub enum Kind {')
    for ob in objects:
        doc = ob.get('doc') or 'Protocols: %s.' % ', '.join('`%s`' % p for p in ob['protocols'])
        w('    /// %s' % doc)
        w('    %s,' % ob['name'])
    w('}')
    w('')
    w('impl Kind {')
    w('    /// Every kind, in schema order.')
    w('    pub const ALL: [Kind; %d] = [%s];' % (len(objects), ', '.join('Kind::' + ob['name'] for ob in objects)))
    w('')
    w('    /// The kind\'s schema name.')
    w('    pub fn name(self) -> &\'static str {')
    w('        match self {')
    for ob in objects:
        w('            Kind::%s => "%s",' % (ob['name'], ob['name']))
    w('        }')
    w('    }')
    w('')
    w('    /// Whether some record creates objects of this kind. Kinds no record creates name sets of other kinds')
    w('    /// (`Object`, `Resource`) or objects the protocol cannot create yet.')
    w('    pub fn is_created(self) -> bool {')
    w('        matches!(self, %s)' % ' | '.join('Kind::' + ob['name'] for ob in objects if ob['name'] in created))
    w('    }')
    w('')
    w('    /// Whether an object of this (created) kind may be passed in a field of kind `field`: the same kind,')
    w('    /// `Object`, or a kind no record creates whose protocols include all of this kind\'s.')
    w('    pub fn satisfies(self, field: Kind) -> bool {')
    w('        match (self, field) {')
    w('            (a, b) if a == b => true,')
    w('            (_, Kind::Object) => true,')
    for e in objects:
        if e['name'] in created or not e['protocols']:
            continue
        members = [k['name'] for k in objects
                   if k['name'] in created and k['protocols'] and set(k['protocols']) <= set(e['protocols'])]
        if members:
            w('            (%s, Kind::%s) => true,' % (' | '.join('Kind::' + m for m in members), e['name']))
    w('            _ => false,')
    w('        }')
    w('    }')
    w('}')
    w('')

    # Scopes -----------------------------------------------------------------------------------------------------
    w('/// Where a command-stream record may appear, from the Metal protocol of the methods it carries.')
    w('#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]')
    w('pub enum Scope {')
    w('    /// Outside any encoder (`MTLCommandBuffer`, `MTLCommandBufferSPI`).')
    w('    CommandBuffer,')
    w('    /// Inside an encoder of any kind (`MTLCommandEncoder`).')
    w('    AnyEncoder,')
    w('    /// Inside a render encoder, including a parallel encoder\'s sub-encoder.')
    w('    Render,')
    w('    /// Inside a compute encoder.')
    w('    Compute,')
    w('    /// Inside a blit encoder.')
    w('    Blit,')
    w('    /// Inside a parallel render encoder, outside its sub-encoders.')
    w('    Parallel,')
    w('}')
    w('')
    w('/// The encoder a `Begin*` record opens.')
    w('#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]')
    w('pub enum Opens {')
    w('    Render,')
    w('    Parallel,')
    w('    /// A render encoder inside a parallel render encoder (`BeginParallelSubEncoder`).')
    w('    ParallelSub,')
    w('    Compute,')
    w('    Blit,')
    w('}')
    w('')
    w('/// The placement rule of one stream opcode.')
    w('#[derive(Clone, Copy, Debug, PartialEq, Eq)]')
    w('pub struct StreamRule {')
    w('    pub scope: Scope,')
    w('    /// The encoder the record opens, for `Begin*` records.')
    w('    pub opens: Option<Opens>,')
    w('}')
    w('')
    w('/// The placement rule of a stream opcode; `None` for an opcode of another channel.')
    w('pub fn stream_rule(opcode: u16) -> Option<StreamRule> {')
    w('    let (scope, opens) = match opcode {')
    for op in schema.by_channel['stream']:
        sc = scope_of(op)
        op_opens = opens_of(op)
        if op_opens == 'ParallelSub' and sc != 'Parallel':
            raise SystemExit('%s: a sub-encoder opens inside a parallel encoder' % op.name)
        if op_opens and op_opens != 'ParallelSub' and sc != 'CommandBuffer':
            raise SystemExit('%s: an encoder opens on the command buffer' % op.name)
        w('        0x%04x => (Scope::%s, %s),' % (op.code, sc, 'Some(Opens::%s)' % op_opens if op_opens else 'None'))
    w('        _ => return None,')
    w('    };')
    w('    Some(StreamRule { scope, opens })')
    w('}')
    w('')

    # Handles ----------------------------------------------------------------------------------------------------
    w('/// What a handle field means to the record.')
    w('#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]')
    w('pub enum Role {')
    w('    /// The handle the guest allocated for the object the record creates.')
    w('    Result,')
    w('    /// The object the Metal method is sent to.')
    w('    Receiver,')
    w('    /// Any other object the record names.')
    w('    Argument,')
    w('    /// The object the guest deallocated (`Release`).')
    w('    Released,')
    w('}')
    w('')
    w('/// One non-nil handle a record carries.')
    w('#[derive(Clone, Copy, Debug, PartialEq, Eq)]')
    w('pub struct HandleUse {')
    w('    /// The field, as a path from the record (`[]` marks an array element).')
    w('    pub field: &\'static str,')
    w('    /// The kind the schema gives the field.')
    w('    pub kind: Kind,')
    w('    pub role: Role,')
    w('    pub handle: Handle,')
    w('}')
    w('')
    vis = Visitor(schema)
    for channel, enum, fn in (('control', 'Request', 'request_handles'), ('stream', 'Command', 'command_handles')):
        what = 'request' if channel == 'control' else 'record'
        w('/// Report every non-nil handle of the %s, in field order.' % what)
        w('pub fn %s(record: &%s, f: &mut dyn FnMut(HandleUse)) {' % (fn, enum))
        w('    match record {')
        for op in schema.by_channel[channel]:
            body = []
            for fl in op.fields:
                if vis.has_handles(fl.type):
                    body += vis.visit(fl.type, 'v.%s' % fl.rust, fl.name, role_of(op, fl), 12, False)
            if body:
                w('        %s::%s(v) => {' % (enum, op.name))
                o.extend(body)
                w('        }')
            else:
                w('        %s::%s(_) => {}' % (enum, op.name))
        w('    }')
        w('}')
        w('')

    # Failure replies --------------------------------------------------------------------------------------------
    w('/// The reply to a control request with this opcode that carries `failure` (whose status is not 0); `None`')
    w('/// for an opcode that is not a control request.')
    w('pub fn failure_reply(opcode: u16, failure: Failure) -> Option<Reply> {')
    w('    Some(match opcode {')
    for op in schema.by_channel['control']:
        w('        0x%04x => Reply::%s(Err(failure)),' % (op.code, op.name))
    w('        _ => return None,')
    w('    })')
    w('}')
    w('')
    w('/// The failure a reply carries, or `None` for a success.')
    w('pub fn reply_failure(reply: &Reply) -> Option<&Failure> {')
    w('    match reply {')
    for op in schema.by_channel['control']:
        w('        Reply::%s(r) => r.as_ref().err(),' % op.name)
    w('    }')
    w('}')
    return '\n'.join(o) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='exit 1 if schema.rs is stale')
    args = parser.parse_args()
    text = gen(Schema(SCHEMA))
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print('stale: %s' % OUT.relative_to(ROOT))
            return 1
        return 0
    OUT.write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
