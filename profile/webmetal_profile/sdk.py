"""Enumerator names and values from the installed iPhoneOS SDK's Metal headers.

The SDK headers are the reference for public names. Values are read from clang's AST of
`<Metal/Metal.h>` (EnumConstantDecl nodes), so they are the compiler's evaluation of the
header, not text matched by pattern.
"""

import functools
import json
import subprocess
import tempfile
from pathlib import Path


class SdkError(Exception):
    pass


def _run(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise SdkError(f'{" ".join(args)} failed: {result.stderr[-2000:]}')
    return result.stdout


@functools.lru_cache(maxsize=None)
def sdk_version():
    return _run(['xcrun', '--sdk', 'iphoneos', '--show-sdk-version']).strip()


@functools.lru_cache(maxsize=None)
def sdk_path():
    return _run(['xcrun', '--sdk', 'iphoneos', '--show-sdk-path']).strip()


def _documents(text):
    decoder, at, out = json.JSONDecoder(), 0, []
    while at < len(text):
        while at < len(text) and text[at].isspace():
            at += 1
        if at >= len(text):
            break
        doc, at = decoder.raw_decode(text, at)
        out.append(doc)
    return out


def _constant_value(node):
    stack = [node]
    while stack:
        n = stack.pop(0)
        if isinstance(n, dict):
            if n.get('kind') == 'ConstantExpr' and 'value' in n:
                return int(n['value'])
            stack.extend(n.get('inner', []))
    return None


@functools.lru_cache(maxsize=None)
def enum_constants(enum_name):
    """[(name, value)] of the enum declared as enum_name in <Metal/Metal.h>, in declaration order."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / 'metal.m'
        source.write_text('#import <Metal/Metal.h>\n')
        text = _run(['xcrun', '--sdk', 'iphoneos', 'clang', '-target', 'arm64-apple-ios17.0',
                     '-x', 'objective-c', '-fsyntax-only', '-Xclang', '-ast-dump=json',
                     '-Xclang', f'-ast-dump-filter={enum_name}', str(source)])
    out, seen = [], set()
    for doc in _documents(text):
        if doc.get('kind') != 'EnumDecl' or doc.get('name') != enum_name:
            continue
        previous = -1
        for node in doc.get('inner', []):
            if node.get('kind') != 'EnumConstantDecl':
                continue
            value = _constant_value(node)
            if value is None:
                # An enumerator without an initializer is the previous one plus one, and the
                # first is zero (C11 6.7.2.2p3); clang's AST records only explicit values.
                value = previous + 1
            previous = value
            if node['name'] in seen:
                continue
            seen.add(node['name'])
            out.append((node['name'], value))
    if not out:
        raise SdkError(f'no enumerators found for {enum_name}')
    return tuple(out)
