#!/usr/bin/env python3
"""Check the WebMetal specification tables.

Fails (exit status 1) when any table cell is empty, any row lacks a citation (open
questions instead need a reason), any citation does not verify against its pinned
source, or a generated table differs from what its generator produces now.

Usage: check.py --cache DIR
"""

import argparse
import difflib
import os
import subprocess
import sys
import tempfile
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(HERE, "gen"))
import citecheck  # noqa: E402
import specdoc  # noqa: E402

GENERATED = [
    ("limits.toml", "gen_limits.py"),
    ("formats.toml", "gen_formats.py"),
]

ROW_ARRAYS = {
    "limits.toml": ["limit", "feature", "wgsl_language_extension", "question"],
    "formats.toml": ["webgpu_texture_format", "webgpu_vertex_format", "metal_pixel_format",
                     "metal_vertex_format", "question"],
}


def load_measures():
    """{measurement id: {case id: text}} from the committed measurement results."""
    out = {}
    mdir = os.path.join(HERE, "measure")
    for name in sorted(os.listdir(mdir)):
        if not name.endswith(".toml"):
            continue
        with open(os.path.join(mdir, name), "rb") as f:
            d = tomllib.load(f)
        cases = {"meta": "\n".join("%s: %s" % (k, v) for k, v in d["meta"].items())}
        for c in d.get("case", []):
            lines = []
            for k, v in c.items():
                lines.append("%s: %s" % (k, " ".join(str(x) for x in v) if isinstance(v, list) else v))
            cases[c["id"]] = "\n".join(lines)
        out[d["meta"]["id"]] = cases
    return out


def regenerate(cache, name, script):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, name)
        subprocess.run([sys.executable, "-B", "-I", os.path.join(HERE, "gen", script), "--cache", cache,
                        "--out", out], check=True, stdout=subprocess.DEVNULL)
        with open(out, encoding="utf-8") as f:
            return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=os.environ.get("TABLES_CACHE"))
    a = ap.parse_args()
    if not a.cache:
        raise SystemExit("a cache directory is required (--cache or TABLES_CACHE)")
    sources = specdoc.load_sources()
    chk = citecheck.Checker(sources, a.cache, measures=load_measures())

    for name, script in GENERATED:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            chk.fail(name, "missing")
            continue
        with open(path, encoding="utf-8") as f:
            have = f.read()
        want = regenerate(a.cache, name, script)
        if have != want:
            diff = "".join(list(difflib.unified_diff(have.splitlines(True), want.splitlines(True),
                                                     name, name + " (regenerated)"))[:40])
            chk.fail(name, "differs from its generator output:\n" + diff)

    for name, arrays in ROW_ARRAYS.items():
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            data = tomllib.load(f)
        chk.no_empty(name + ".meta", data.get("meta", {}))
        for arr in arrays:
            for i, row in enumerate(data.get(arr, [])):
                label = row.get("id") or row.get("name") or str(i)
                chk.row("%s %s[%s]" % (name, arr, label), row)

    refusal_src = os.path.join(HERE, "gen", "refusals.toml")
    if os.path.exists(refusal_src):
        with open(refusal_src, "rb") as f:
            data = tomllib.load(f)
        for i, row in enumerate(data.get("refusal", [])):
            chk.row("refusals.toml refusal[%s]" % row.get("id", i), row)

    for e in chk.errors:
        print("FAIL " + e)
    print("%d citations checked, %d failures" % (chk.checked, len(chk.errors)))
    return 1 if chk.errors else 0


if __name__ == "__main__":
    sys.exit(main())
