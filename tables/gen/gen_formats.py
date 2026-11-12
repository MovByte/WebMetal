#!/usr/bin/env python3
"""Generate tables/formats.toml.

WebGPU side, from the pinned WebGPU text: the GPUTextureFormat and GPUVertexFormat IDL
enums, joined with the texture format capability tables (section 26.1) and the vertex
format table (section 10.3.7.1); every IDL value must have exactly one table row.
Metal side: the MTLPixelFormat and MTLVertexFormat enums of the pinned iPhoneOS SDK
header, each value's availability on iOS 17.7 evaluated from its annotation, and
Apple's description of each format from the pinned documentation page. The pairing
of the two sides comes from gen/format_map.toml; every Metal format available on
iOS 17.7 must be placed in exactly one group. Questions are merged from
gen/formats_questions.toml.

Usage: gen_formats.py --cache DIR [--out PATH]
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
TARGET_IOS = (17, 7)

PIXEL_HEADER = "Metal.framework/Headers/MTLPixelFormat.h"
VERTEX_HEADER = "Metal.framework/Headers/MTLVertexDescriptor.h"
# A capability cell the specification leaves empty (the capability is absent).
BLANK = "(empty)"
CAP_SECTIONS = [("plain-color-formats", "plain"), ("depth-formats", "depth-stencil"),
                ("packed-formats", "packed")]


def version(s):
    return tuple(int(x) for x in s.split("."))


def parse_enum(text, enum, prefix):
    """(name, value, annotation, source line) for each enumerator, comments removed."""
    start = text.index("typedef NS_ENUM(NSUInteger, %s)" % enum)
    body = text[start:]
    end = body.index("} API_AVAILABLE")
    enum_avail = re.match(r"\} (API_AVAILABLE\([^;]*\));", body[end:]).group(1)
    body = body[:end]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    out = []
    pat = r"(%s\w+)\s*((?:API_\w+\((?:[^()]|\([^()]*\))*\)\s*)*)=\s*(\d+)\s*," % prefix
    for m in re.finditer(pat, body):
        annot = " ".join(m.group(2).split())
        line = "%s %s= %s," % (m.group(1), annot + " " if annot else "", m.group(3))
        out.append((m.group(1), int(m.group(3)), annot, " ".join(line.split())))
    return out, enum_avail


def ios_availability(annot, enum_avail):
    """Availability of an enumerator on iOS 17.7 and the annotation it follows from."""
    if re.search(r"API_UNAVAILABLE\([^)]*\bios\b", annot):
        return False, "unavailable on iOS"
    m = re.search(r"API_DEPRECATED\(.*?\bios\(([\d.]+),\s*([\d.]+)\)", annot)
    if m:
        intro, dep = m.group(1), m.group(2)
        ok = version(intro) <= TARGET_IOS
        return ok, "introduced in iOS %s, deprecated in iOS %s" % (intro, dep)
    m = re.search(r"API_AVAILABLE\(.*?\bios\(([\d.]+)\)", annot)
    if m:
        intro = m.group(1)
        return version(intro) <= TARGET_IOS, "introduced in iOS %s" % intro
    m = re.search(r"\bios\(([\d.]+)\)", enum_avail)
    return version(m.group(1)) <= TARGET_IOS, "enum available since iOS %s" % m.group(1)


def apple_abstracts(path):
    import json
    with open(path, "rb") as f:
        d = json.load(f)
    out = {}
    for ref in d.get("references", {}).values():
        if ref.get("kind") != "symbol":
            continue
        nav = ref.get("navigatorTitle") or []
        if not nav or "text" not in nav[0]:
            continue
        out[nav[0]["text"]] = specdoc._apple_inline(ref.get("abstract"))
    return out


def raw_row_quote(row):
    return " | ".join(specdoc.norm(c["text"]) for c in row)


def section_tables(doc, section_id):
    h = doc.heading_for(section_id)
    start = h["offset"]
    end = len(doc.text)
    for other in doc.headings:
        if other["offset"] > start and other["level"] <= h["level"]:
            end = other["offset"]
            break
    return [t for t in doc.tables if start <= t.offset < end]


def webgpu_texture_formats(doc, idl):
    names = specdoc.idl_enum(idl, "GPUTextureFormat")
    rows = {}
    for section, kind in CAP_SECTIONS:
        for t in section_tables(doc, section):
            grid = t.grid()
            header = grid[0]
            if header[0]["text"] != "Format":
                continue
            # A second header row names the STORAGE_BINDING access columns.
            body_start = 1
            cols = [c["text"] for c in header]
            if len(grid) > 1 and grid[1][0]["text"] == "Format":
                cols = ["%s %s" % (a["text"], b["text"]) if a is not b else a["text"]
                        for a, b in zip(header, grid[1])]
                body_start = 2
            for raw, g in zip(t.rows[body_start:], grid[body_start:]):
                fmt = g[0]["text"]
                if fmt not in names:
                    continue
                caps = {}
                for col, cell in zip(cols[1:], g[1:]):
                    caps[col] = cell["text"] or BLANK
                entry = rows.setdefault(fmt, {"name": fmt, "table": kind, "aspects": [],
                                              "cite": [{"src": "webgpu", "ref": "#enumdef-gputextureformat",
                                                        "quote": '"%s"' % fmt}]})
                entry["aspects"].append(caps)
                entry["cite"].append({"src": "webgpu", "ref": "#" + section, "quote": raw_row_quote(raw)})
    missing = [n for n in names if n not in rows]
    if missing:
        raise SystemExit("texture formats without a capability row: %r" % missing)
    return [rows[n] for n in names]


def webgpu_vertex_formats(doc, idl):
    names = specdoc.idl_enum(idl, "GPUVertexFormat")
    rows = {}
    for t in section_tables(doc, "vertex-formats"):
        grid = t.grid()
        if grid[0][0]["text"] != "Vertex format":
            continue
        cols = [c["text"] for c in grid[0]]
        for raw, g in zip(t.rows[1:], grid[1:]):
            fmt = g[0]["text"].strip('"')
            if fmt not in names:
                continue
            rows[fmt] = {
                "name": fmt,
                "data_type": g[cols.index("Data type")]["text"],
                "components": int(g[cols.index("Components")]["text"]),
                "byte_size": int(g[cols.index("byteSize")]["text"]),
                "cite": [
                    {"src": "webgpu", "ref": "#enumdef-gpuvertexformat", "quote": '"%s"' % fmt},
                    {"src": "webgpu", "ref": "#vertex-formats", "quote": raw_row_quote(raw)},
                ],
            }
    missing = [n for n in names if n not in rows]
    if missing:
        raise SystemExit("vertex formats without a table row: %r" % missing)
    return [rows[n] for n in names]


def metal_formats(sources, header, enum, prefix, abstracts, page_src, groups, webgpu_names):
    text = specdoc.sdk_document("sdk", sources, header).text
    entries, enum_avail = parse_enum(text, enum, prefix)
    placed = {}
    for g in groups:
        for pair in g.get("pairs", []):
            placed.setdefault(pair[0], []).append((g, pair[1], None))
        for f in g.get("formats", []):
            placed.setdefault(f[0], []).append((g, None, f[1]))
    names = {e[0] for e in entries}
    unknown = sorted(set(placed) - names)
    if unknown:
        raise SystemExit("format_map names formats the header lacks: %r" % unknown)
    out = []
    for name, value, annot, line in entries:
        ok, why = ios_availability(annot, enum_avail)
        row = {
            "name": name,
            "value": value,
            "ios_17_7": ok,
            "availability": why,
        }
        cites = [{"sdk": header, "quote": line}]
        if not annot:
            cites.append({"sdk": header, "quote": "} %s;" % enum_avail})
        desc = abstracts.get(name)
        if desc:
            row["description"] = desc
            cites.append({"src": page_src, "quote": "%s: %s" % (name, desc)})
        where = placed.get(name, [])
        if not ok:
            if where:
                raise SystemExit("%s is not in the iOS 17.7 API but format_map places it" % name)
            row["class"] = "not-in-ios-17.7"
            row["cite"] = cites
            out.append(row)
            continue
        if len(where) != 1:
            raise SystemExit("%s must be in exactly one group of format_map (found %d)" % (name, len(where)))
        g, webgpu, plan_row = where[0]
        row["class"] = g["id"]
        if webgpu is not None:
            if webgpu not in webgpu_names:
                raise SystemExit("%s pairs with unknown WebGPU format %s" % (name, webgpu))
            row["webgpu"] = webgpu
        if plan_row is not None and plan_row != "none":
            row["plan_row"] = plan_row
        row["basis"] = g["basis"]
        row["cite"] = cites + g["cite"]
        if webgpu is not None:
            row["cite"].append({"src": "webgpu", "ref": "#enumdef-gputextureformat" if prefix == "MTLPixelFormat"
                                else "#enumdef-gpuvertexformat", "quote": '"%s"' % webgpu})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=os.environ.get("TABLES_CACHE"))
    ap.add_argument("--out", default=os.path.join(TABLES, "formats.toml"))
    a = ap.parse_args()
    sources = specdoc.load_sources()
    webgpu = specdoc.document("webgpu", sources, a.cache)
    idl = specdoc.idl_text(webgpu)
    with open(os.path.join(HERE, "format_map.toml"), "rb") as f:
        fmap = tomllib.load(f)
    with open(os.path.join(HERE, "formats_questions.toml"), "rb") as f:
        questions = tomllib.load(f).get("question", [])

    tex = webgpu_texture_formats(webgpu, idl)
    vtx = webgpu_vertex_formats(webgpu, idl)
    pix_abs = apple_abstracts(specdoc.fetch("apple-mtlpixelformat", sources, a.cache))
    vtx_abs = apple_abstracts(specdoc.fetch("apple-mtlvertexformat", sources, a.cache))
    pix = metal_formats(sources, PIXEL_HEADER, "MTLPixelFormat", "MTLPixelFormat", pix_abs,
                        "apple-mtlpixelformat", fmap["pixel_group"], {r["name"] for r in tex})
    vfm = metal_formats(sources, VERTEX_HEADER, "MTLVertexFormat", "MTLVertexFormat", vtx_abs,
                        "apple-mtlvertexformat", fmap["vertex_group"], {r["name"] for r in vtx})

    parts = [
        "# Generated by tables/gen/gen_formats.py from the sources pinned in tables/sources.toml,\n"
        "# tables/gen/format_map.toml and tables/gen/formats_questions.toml. Do not edit by hand.\n",
        tomlout.table("meta", {
            "webgpu": "%s, %s %s, %s" % (sources["webgpu"]["title"], sources["webgpu"]["version"],
                                         sources["webgpu"]["date"], sources["webgpu"]["url"]),
            "metal_headers": sources["sdk"]["title"],
            "metal_descriptions": "Apple Developer Documentation pages MTLPixelFormat and MTLVertexFormat, retrieved %s"
                                  % sources["apple-mtlpixelformat"]["date"],
            "target": "iOS %d.%d: a Metal format is in scope when its header annotation makes it available on that version" % TARGET_IOS,
            "device_support": "Which formats and capabilities the A10X supports comes from the D1 device profile, not from this table",
            "private_formats": "The private YCbCr formats are not in the SDK header; their list comes from D1",
            "empty_cells": "A WebGPU capability cell that the specification table leaves empty is written %s" % BLANK,
        }),
    ]
    for r in tex:
        parts.append(tomlout.array_item("webgpu_texture_format", r))
    for r in vtx:
        parts.append(tomlout.array_item("webgpu_vertex_format", r))
    for r in pix:
        parts.append(tomlout.array_item("metal_pixel_format", r))
    for r in vfm:
        parts.append(tomlout.array_item("metal_vertex_format", r))
    for q in questions:
        parts.append(tomlout.array_item("question", q))
    with open(a.out, "w") as f:
        f.write("\n".join(parts))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
