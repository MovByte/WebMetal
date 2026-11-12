"""Method B2: driver getters decoded statically with LLVM's disassembler.

A driver getter that is a straight-line sequence of immediate moves into the return
registers followed by `ret` is decoded from `llvm-mc --disassemble` text, independently of
the emulator: a different instruction decoder (LLVM's, not QEMU's) and an evaluator that
understands only these forms. Anything else in the body (a load, a call, a compare) makes
the getter unreachable for this method, and that is reported rather than approximated.
"""

import re
import shutil
import struct
import subprocess
from pathlib import Path

MASK64 = (1 << 64) - 1


class DecodeError(Exception):
    pass


def llvm_mc(llvm_bin=None):
    if llvm_bin:
        path = Path(llvm_bin) / 'llvm-mc'
        if not path.exists():
            raise DecodeError(f'no llvm-mc in {llvm_bin}')
        return str(path)
    found = shutil.which('llvm-mc')
    if not found:
        raise DecodeError('llvm-mc not found; pass the LLVM bin directory')
    return found


def disassemble(tool, code):
    """[(mnemonic, operands)] for the little-endian AArch64 words in code."""
    text = ' '.join(f'0x{b:02x}' for b in code)
    result = subprocess.run([tool, '--disassemble', '-triple=arm64-apple-ios', '-show-encoding'],
                            input=text, capture_output=True, text=True, timeout=120)
    if result.returncode or 'invalid instruction' in result.stderr:
        raise DecodeError(f'llvm-mc failed: {result.stderr.strip()}')
    out = []
    for line in result.stdout.splitlines():
        line = line.split(';')[0].strip()
        if not line or line.startswith('.'):
            continue
        parts = line.split(None, 1)
        out.append((parts[0], parts[1] if len(parts) > 1 else ''))
    if len(out) != len(code) // 4:
        raise DecodeError('llvm-mc instruction count differs from input')
    return out


def _imm(text):
    text = text.strip().lstrip('#')
    return int(text, 0)


def evaluate(instructions, state=None):
    """Value(s) a straight-line constant getter returns, or raise DecodeError.

    Returns {'x0': v, 'x1': v, 's0': bits, 'x8_stores': {offset: (size, value)}} for the
    registers written before the first `ret`.
    """
    # Arguments after self and _cmd arrive in x2..x7; they stay symbolic.
    if state is None:
        state = ({f'x{i}': ('arg', i - 2) for i in range(2, 8)}, {}, {})
    regs, vectors, stores = state
    for index, (mnemonic, ops) in enumerate(instructions):
        args = [a.strip() for a in re.split(r',(?![^\[]*\])', ops)] if ops else []
        if mnemonic == 'b' and len(args) == 1 and index == len(instructions) - 1:
            # A tail branch: the caller continues decoding at the target with this state.
            return {'branch': _imm(args[0]), 'state': (regs, vectors, stores)}
        if mnemonic == 'ret' and not args:
            out = {}
            for name in ('x0', 'x1', 's0'):
                if name in regs:
                    out[name] = regs[name]
            if stores:
                out['x8_stores'] = stores
            return out
        if mnemonic == 'mov' and len(args) == 2 and args[1].startswith('#'):
            reg, value = args[0], _imm(args[1])
            if reg[0] == 'w':
                regs['x' + reg[1:]] = value & 0xffffffff
            elif reg[0] == 'x':
                regs[reg] = value & MASK64
            else:
                raise DecodeError(f'unhandled {mnemonic} {ops}')
            continue
        if mnemonic == 'mov' and len(args) == 2 and args[1] in ('wzr', 'xzr'):
            regs['x' + args[0][1:]] = 0
            continue
        if mnemonic == 'mov' and len(args) == 2 and re.fullmatch(r'x\d+', args[0]) and re.fullmatch(r'x\d+', args[1]):
            if args[1] not in regs:
                raise DecodeError(f'mov from undefined {args[1]}')
            regs[args[0]] = regs[args[1]]
            continue
        if mnemonic == 'dup' and len(args) == 2 and args[0].endswith('.2d') and args[1] in regs:
            vectors[args[0].split('.')[0][1:]] = (regs[args[1]], regs[args[1]])
            continue
        if mnemonic == 'movk' and len(args) in (2, 3):
            reg, value = args[0], _imm(args[1])
            shift = int(re.fullmatch(r'lsl #(\d+)', args[2]).group(1)) if len(args) == 3 else 0
            key = 'x' + reg[1:]
            if key not in regs:
                raise DecodeError(f'movk into undefined {reg}')
            width = 32 if reg[0] == 'w' else 64
            current = regs[key] & ~(0xffff << shift)
            regs[key] = (current | value << shift) & ((1 << width) - 1)
            continue
        if mnemonic == 'fmov' and len(args) == 2 and args[0].startswith('s'):
            if args[1].startswith('#'):
                regs[args[0]] = struct.unpack('<I', struct.pack('<f', float(args[1].lstrip('#'))))[0]
                continue
            if args[1].startswith('w') and 'x' + args[1][1:] in regs:
                regs[args[0]] = regs['x' + args[1][1:]] & 0xffffffff
                continue
        if mnemonic in ('str', 'stp') and args and args[-1].startswith('[x8'):
            match = re.fullmatch(r'\[x8(?:, #(-?\w+))?\]', args[-1])
            if not match:
                raise DecodeError(f'unhandled store {mnemonic} {ops}')
            offset = int(match.group(1), 0) if match.group(1) else 0
            sources = args[:-1]
            if len(sources) == 1 and sources[0].startswith('q') and sources[0][1:] in vectors:
                low, high = vectors[sources[0][1:]]
                stores[offset] = (8, low)
                stores[offset + 8] = (8, high)
                continue
            for i, src in enumerate(sources):
                size = 8 if src[0] == 'x' else 4
                if src in ('xzr', 'wzr'):
                    value = 0
                elif 'x' + src[1:] in regs:
                    value = regs['x' + src[1:]] & ((1 << 8 * size) - 1)
                else:
                    raise DecodeError(f'store of undefined {src}')
                stores[offset + i * size] = (size, value)
            continue
        raise DecodeError(f'not a constant getter: {mnemonic} {ops}')
    raise DecodeError('no ret within the decoded range')


def stub_selector(cache, tool, address):
    """Selector an `_objc_msgSend$<selector>` stub loads into x1 (adrp x1 + ldr x1)."""
    (m1, ops1), (m2, ops2) = disassemble(tool, cache.read(address, 8))
    page = re.fullmatch(r'x1, #(-?\d+)', ops1)
    load = re.fullmatch(r'x1, \[x1, #(\d+)\]', ops2)
    if m1 != 'adrp' or not page or m2 != 'ldr' or not load:
        raise DecodeError(f'{address:#x} is not a selector stub')
    selref = (address & ~0xfff) + int(page.group(1)) + int(load.group(1))
    return cache.cstring(cache.pointer(selref))


def in_section(cache, address, name):
    image = cache.image_at(address)
    return image is not None and any(start <= address < start + size
                                     for _, start, size in image.section('__TEXT', name))


def decode_getter(cache, tool, address, resolve=None, limit=16, hops=4):
    """Decode a getter at address, following unconditional tail branches.

    A tail branch to an `_objc_msgSend$<selector>` stub is a message to self with the same
    arguments; `resolve(selector)` names the implementation the receiver's class has for it.
    """
    state = None
    for _ in range(hops):
        if in_section(cache, address, '__objc_stubs'):
            if resolve is None:
                raise DecodeError('tail call through a selector stub')
            address = resolve(stub_selector(cache, tool, address))
            continue
        image = cache.image_at(address)
        if image is None:
            raise DecodeError(f'{address:#x} is not in an image')
        name, offset = image.symbolize(address)
        if offset:
            raise DecodeError(f'{address:#x} is not a symbol start')
        code = cache.read(address, 4 * limit)
        words = struct.unpack(f'<{limit}I', code)
        end = next((i for i, w in enumerate(words) if w == 0xd65f03c0 or w & 0xfc000000 == 0x14000000), None)
        if end is None:
            raise DecodeError('no ret or branch in the first instructions')
        result = evaluate(disassemble(tool, code[:4 * (end + 1)]), state)
        if 'branch' not in result:
            return result
        address = address + 4 * end + result['branch']
        state = result['state']
    raise DecodeError('tail branch chain too long')
