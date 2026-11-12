#!/usr/bin/env python3
"""Generate tables/limits.toml from the pinned WebGPU and WGSL specifications.

Limits: the attribute list of the `GPUSupportedLimits` IDL interface, joined with
the rows of the specification's limits table (type, limit class, default and
compatibility-mode default). The two lists must name the same limits.
Features: the `GPUFeatureName` IDL enum, with each feature's index section, the
WGSL enable-extension that it gates, and its dependencies.
WGSL language extensions: the WGSL language-extension table.
Resolved and open questions are merged in from gen/limits_questions.toml.

Usage: gen_limits.py --cache DIR [--out PATH]
"""

import argparse
import os
import re
import sys
import tomllib

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specdoc  # noqa: E402
import tomlout  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)


def table_after(doc, anchor_id):
    for t in doc.tables:
        if t.anchor == anchor_id:
            return t
    raise SystemExit("table %s not found" % anchor_id)


def section_table(doc, section_id, header0):
    """The first table in a section whose first header cell is header0."""
    scope_start = doc.anchors[section_id]
    h = doc.heading_for(section_id)
    end = len(doc.text)
    for other in doc.headings:
        if other["offset"] > scope_start and other["level"] <= h["level"]:
            end = other["offset"]
            break
    for t in doc.tables:
        if scope_start <= t.offset < end:
            g = t.grid()
            if g and g[0] and g[0][0]["text"] == header0:
                return t
    raise SystemExit("no table headed %r in %s" % (header0, section_id))


def row_quote(cells):
    """The row's text as it reads in the document: a spanned cell appears once."""
    seen = []
    for c in cells:
        if not any(c is x for x in seen):
            seen.append(c)
    return " | ".join(c["text"] for c in seen)


VALUE_RE = re.compile(r"^(\d+)(?: (bytes)(?: \((\d+) MiB\))?)?$")


def parse_value(name, text):
    """A limit cell is a count or a byte size, optionally restated in MiB."""
    m = VALUE_RE.match(text)
    if not m:
        raise SystemExit("%s: unrecognized limit value %r" % (name, text))
    value = int(m.group(1))
    if m.group(3) is not None and int(m.group(3)) * 1024 * 1024 != value:
        raise SystemExit("%s: %r restates the value inconsistently" % (name, text))
    return value, ("bytes" if m.group(2) else "count")


def limits(webgpu, idl):
    attrs = specdoc.idl_interface_attributes(idl, "GPUSupportedLimits")
    t = table_after(webgpu, "supported-limits")
    grid = t.grid()
    header = [c["text"] for c in grid[0]]
    want = ["Limit name", "Type", "Limit class", "Default", "Compatibility Mode Default"]
    if header != want:
        raise SystemExit("limits table header changed: %r" % header)
    rows = {}
    last = None
    for r in grid[1:]:
        ident = next((i for i in r[0]["ids"] if i.startswith("dom-supported-limits-")), None)
        if ident:
            name, typ, cls, default, compat = [c["text"] for c in r[:5]]
            rows[name] = {"anchor": ident, "type": typ, "class": cls, "default": default,
                          "compat": compat, "row": row_quote(r[:5])}
            last = name
        elif last and r and r[0]["colspan"] == 5:
            rows[last]["description"] = r[0]["text"]
    idl_names = [n for _, n in attrs]
    if sorted(idl_names) != sorted(rows):
        raise SystemExit("IDL GPUSupportedLimits and the limits table disagree: %r / %r" % (
            sorted(set(idl_names) ^ set(rows)), None))
    out = []
    for idl_type, name in attrs:
        r = rows[name]
        default, unit = parse_value(name, r["default"])
        compat, unit2 = parse_value(name, r["compat"])
        if unit != unit2:
            raise SystemExit("%s: default and compatibility default have different units" % name)
        out.append({
            "name": name,
            "idl_type": idl_type,
            "type": r["type"],
            "class": r["class"],
            "default": default,
            "compat_default": compat,
            "unit": unit,
            "description": r.get("description", ""),
            "cite": [
                {"src": "webgpu", "ref": "#" + r["anchor"], "quote": r["row"]},
                {"src": "webgpu", "ref": "#gpusupportedlimits",
                 "quote": "readonly attribute %s %s;" % (idl_type, name)},
            ],
        })
    return out


def feature_section(webgpu, name):
    for h in webgpu.headings:
        if re.match(r'^25\.\d+\. "%s"$' % re.escape(name), h["title"]):
            return h
    raise SystemExit("feature index section for %s not found" % name)


def features(webgpu, wgsl, idl):
    names = specdoc.idl_enum(idl, "GPUFeatureName")
    ext = section_table(wgsl, "enable-extensions-sec", "WGSL enable-extension")
    enables = {}
    for r in ext.grid()[1:]:
        enables[r[1]["text"].strip('"')] = (r[0]["text"], row_quote(r[:2]))
    deps = section_table(webgpu, "feature-dependencies", "This feature...")
    requires = {}
    for r in deps.grid()[1:]:
        requires[r[0]["text"].strip('"')] = (
            [x.strip().strip('"') for x in r[1]["text"].split(",") if x.strip()],
            row_quote(r[:2]))
    out = []
    for name in names:
        h = feature_section(webgpu, name)
        item = {
            "name": name,
            "section": h["title"],
            "cite": [
                {"src": "webgpu", "ref": "#gpufeaturename", "quote": '"%s"' % name},
                {"src": "webgpu", "ref": "#" + h["id"], "quote": h["title"]},
            ],
        }
        if name in enables:
            item["wgsl_enable"] = enables[name][0]
            item["cite"].append({"src": "wgsl", "ref": "#enable-extensions-sec", "quote": enables[name][1]})
        if name in requires:
            item["requires"] = requires[name][0]
            item["cite"].append({"src": "webgpu", "ref": "#feature-dependencies", "quote": requires[name][1]})
        out.append(item)
    return out


def language_extensions(wgsl):
    t = section_table(wgsl, "language-extensions-sec", "WGSL language extension")
    out = []
    for r in t.grid()[1:]:
        out.append({
            "name": r[0]["text"],
            "description": r[1]["text"],
            "cite": [{"src": "wgsl", "ref": "#language-extensions-sec", "quote": r[0]["text"]}],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=os.environ.get("TABLES_CACHE"))
    ap.add_argument("--out", default=os.path.join(TABLES, "limits.toml"))
    a = ap.parse_args()
    sources = specdoc.load_sources()
    webgpu = specdoc.document("webgpu", sources, a.cache)
    wgsl = specdoc.document("wgsl", sources, a.cache)
    idl = specdoc.idl_text(webgpu)
    with open(os.path.join(HERE, "limits_questions.toml"), "rb") as f:
        questions = tomllib.load(f).get("question", [])

    parts = [
        "# Generated by tables/gen/gen_limits.py from the sources pinned in tables/sources.toml\n"
        "# and the questions in tables/gen/limits_questions.toml. Do not edit by hand.\n",
        tomlout.table("meta", {
            "webgpu": "%s, %s %s, %s" % (sources["webgpu"]["title"], sources["webgpu"]["version"],
                                         sources["webgpu"]["date"], sources["webgpu"]["url"]),
            "wgsl": "%s, %s %s, %s" % (sources["wgsl"]["title"], sources["wgsl"]["version"],
                                       sources["wgsl"]["date"], sources["wgsl"]["url"]),
            "limits_from": "IDL interface GPUSupportedLimits joined with the table at #supported-limits",
            "features_from": "IDL enum GPUFeatureName, feature index sections, #feature-dependencies, WGSL #enable-extensions",
        }),
    ]
    for item in limits(webgpu, idl):
        parts.append(tomlout.array_item("limit", item))
    for item in features(webgpu, wgsl, idl):
        parts.append(tomlout.array_item("feature", item))
    for item in language_extensions(wgsl):
        parts.append(tomlout.array_item("wgsl_language_extension", item))
    for q in questions:
        parts.append(tomlout.array_item("question", q))
    with open(a.out, "w") as f:
        f.write("\n".join(parts))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
