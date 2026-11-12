#!/usr/bin/env python3
"""Compare Metal reference harness runs under the recorded comparison rules.

    python3 compare.py REFERENCE CANDIDATE [--only PREFIXES] [--verbose]
    python3 compare.py --check RUN

REFERENCE and CANDIDATE are directories written by `wmref run`. The first
form compares every case of REFERENCE with the same case and variant in
CANDIDATE. The second form checks one run on its own: file digests, record
structure, and that each bounded or property record holds for the run's own
values. FORMAT.md defines the files and the rules.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

FORMAT = "webmetal-reference/1"
OUTPUT_KINDS = {"value", "data", "error", "call"}
RULES = {"exact", "device", "none", "bounded", "property"}
ELEMENTS = {"u8": "<B", "u16": "<H", "u32": "<I", "u64": "<Q", "i8": "<b", "i16": "<h", "i32": "<i",
            "i64": "<q", "f16": "<e", "f32": "<f", "f64": "<d"}


class Run:
    def __init__(self, path):
        self.path = Path(path)
        manifest = self.path / "run.json"
        if not manifest.is_file():
            raise SystemExit(f"{self.path}: no run.json")
        self.manifest = json.loads(manifest.read_text())
        if self.manifest.get("format") != FORMAT:
            raise SystemExit(f"{manifest}: format is not {FORMAT}")
        self.entries = {(e["case"], e["variant"]): e for e in self.manifest["cases"]}

    @property
    def device_key(self):
        return self.manifest["device"]["key"]

    def case(self, identifier, variant):
        path = self.path / variant / f"{identifier}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def digest(self, identifier, variant):
        path = self.path / variant / f"{identifier}.json"
        return hashlib.sha256(path.read_bytes()).hexdigest()


def elements(record):
    data = bytes.fromhex(record["hex"])
    fmt = ELEMENTS[record["elem"]]
    size = struct.calcsize(fmt)
    return [struct.unpack_from(fmt, data, i * size)[0] for i in range(len(data) // size)]


def ideal_values(rule):
    data = bytes.fromhex(rule["ideal"])
    return [struct.unpack_from("<d", data, i * 8)[0] for i in range(len(data) // 8)]


def neighbours(x, elem):
    """The finite values of type elem (f16 or f32) immediately below and above x."""
    fmt = {"f16": "<e", "f32": "<f"}[elem]
    bits_fmt = {"f16": "<H", "f32": "<I"}[elem]
    width = {"f16": 16, "f32": 32}[elem]

    def to_bits(v):
        return struct.unpack(bits_fmt, struct.pack(fmt, v))[0]

    def from_bits(b):
        return struct.unpack(fmt, struct.pack(bits_fmt, b))[0]

    def next_up(b):
        sign = b >> (width - 1)
        magnitude = b & ((1 << (width - 1)) - 1)
        if sign and magnitude == 0:
            return 1
        return b - 1 if sign else b + 1

    def next_down(b):
        sign = b >> (width - 1)
        magnitude = b & ((1 << (width - 1)) - 1)
        if not sign and magnitude == 0:
            return (1 << (width - 1)) | 1
        return b + 1 if sign else b - 1

    try:
        rounded = from_bits(to_bits(x))
    except OverflowError:
        return None
    if rounded == x:
        b = to_bits(x)
        return from_bits(next_down(b)), x, from_bits(next_up(b))
    lower = rounded if rounded < x else from_bits(next_down(to_bits(rounded)))
    upper = rounded if rounded > x else from_bits(next_up(to_bits(rounded)))
    return lower, None, upper


def ulp(x, elem):
    """ulp(x) as defined in MSL 4.1 section 8.4 for binary16 or binary32."""
    if elem == "f64":
        return math.ulp(x)
    found = neighbours(x, elem)
    if found is None:
        return math.inf
    lower, exact, upper = found
    if exact is None:
        return upper - lower
    return min(exact - lower, upper - exact)


def srgb_code(value):
    """Unrounded 8-bit sRGB code of a linear value (MSL 4.1 section 8.7.7)."""
    c = min(max(value, 0.0), 1.0)
    c = 12.92 * c if c < 0.0031308 else 1.055 * math.pow(c, 1.0 / 2.4) - 0.055
    return c * 255.0


def bounded_failures(record):
    rule = record["rule"]
    measured = elements(record)
    ideal = ideal_values(rule)
    if len(measured) != len(ideal):
        return [f"{len(measured)} values for {len(ideal)} ideal values"]
    bound = float(rule["bound"])
    failures = []
    for i, (value, target) in enumerate(zip(measured, ideal)):
        if math.isnan(target):
            ok = math.isnan(value)
        elif rule["model"] == "ulp":
            ok = abs(value - target) <= bound * ulp(target, record["elem"])
        elif rule["model"] == "abs":
            ok = abs(value - target) <= bound
        elif rule["model"] == "srgb8-decode":
            ok = abs(srgb_code(value) - target) <= bound
        else:
            return [f"unknown model {rule['model']}"]
        if not ok:
            failures.append(f"element {i}: {value!r} against ideal {target!r} ({rule['model']} {rule['bound']})")
    return failures


def property_failures(record):
    rule = record["rule"]
    if rule.get("property") != "one-of":
        return [f"unknown property {rule.get('property')}"]
    value = record.get("value")
    return [] if value in rule["values"] else [f"{value!r} is not one of {rule['values']!r}"]


def payload(record):
    """The fields an exact rule compares, per record kind."""
    kind = record["kind"]
    if kind == "value":
        return record["type"], record["value"]
    if kind == "data":
        return record["elem"], record["count"], record["hex"], record.get("layout")
    if kind == "error":
        return record["nil"], record.get("domain"), record.get("code")
    if kind == "call":
        return record["raised"], record.get("exception")
    raise ValueError(kind)


def shape(record):
    kind = record["kind"]
    if kind == "value":
        return kind, record["type"]
    if kind == "data":
        return kind, record["elem"], record.get("layout")
    return (kind,)


def check_case(document, problems, where):
    """Structure and self-consistency of one case file."""
    for key in ("format", "case", "area", "title", "variant", "status", "outcome", "records"):
        if key not in document:
            problems.append(f"{where}: missing {key}")
            return
    if document["format"] != FORMAT:
        problems.append(f"{where}: format {document['format']}")
    if document["status"] == "harness-error":
        problems.append(f"{where}: harness-error: {document.get('message')}")
    names = set()
    for record in document["records"]:
        kind = record.get("kind")
        if kind in ("step", "require"):
            continue
        if kind not in OUTPUT_KINDS:
            problems.append(f"{where}: unknown record kind {kind}")
            continue
        name = record.get("name")
        if name in names:
            problems.append(f"{where}: output {name} recorded twice")
        names.add(name)
        rule = record.get("rule", {})
        compare = rule.get("compare")
        if compare not in RULES:
            problems.append(f"{where}: {name}: unknown rule {compare}")
        elif compare == "none" and not rule.get("reason"):
            problems.append(f"{where}: {name}: rule none without a reason")
        elif compare == "bounded":
            for failure in bounded_failures(record):
                problems.append(f"{where}: {name}: {failure}")
        elif compare == "property":
            for failure in property_failures(record):
                problems.append(f"{where}: {name}: {failure}")


def compare_case(ref, cand, same_profile, notes):
    """Returns (verdict, messages); verdict is pass, fail or incomparable."""
    messages = []
    ref_req = [(r["name"], r["value"]) for r in ref["records"] if r["kind"] == "require"]
    cand_req = [(r["name"], r["value"]) for r in cand["records"] if r["kind"] == "require"]
    if ref_req != cand_req:
        message = f"requirements differ: {ref_req} / {cand_req}"
        return ("fail" if same_profile else "incomparable"), [message]
    if ref["status"] != cand["status"]:
        messages.append(f"status {ref['status']} / {cand['status']}")
    ro, co = ref["outcome"], cand["outcome"]
    if (ro["kind"], ro.get("code"), ro.get("signal")) != (co["kind"], co.get("code"), co.get("signal")):
        messages.append(f"outcome {ro} / {co}")
    elif ro.get("assertion") != co.get("assertion"):
        notes.append(f"{ref['variant']}/{ref['case']}: assertion text differs (not compared)")
    ref_steps = [r["name"] for r in ref["records"] if r["kind"] == "step"]
    cand_steps = [r["name"] for r in cand["records"] if r["kind"] == "step"]
    if ref_steps != cand_steps:
        messages.append(f"steps {ref_steps} / {cand_steps}")
    ref_out = {r["name"]: r for r in ref["records"] if r["kind"] in OUTPUT_KINDS}
    cand_out = {r["name"]: r for r in cand["records"] if r["kind"] in OUTPUT_KINDS}
    for name in cand_out.keys() - ref_out.keys():
        messages.append(f"{name}: not in the reference")
    for name, r in ref_out.items():
        c = cand_out.get(name)
        if c is None:
            messages.append(f"{name}: missing")
            continue
        if c.get("rule") != r.get("rule") and r["rule"]["compare"] != "bounded":
            messages.append(f"{name}: rule differs")
            continue
        compare = r["rule"]["compare"]
        if shape(c) != shape(r):
            messages.append(f"{name}: kind or type differs")
        elif compare == "exact" or (compare == "device" and same_profile):
            if payload(c) != payload(r):
                messages.append(f"{name}: differs")
        elif compare == "bounded":
            if c["rule"].get("ideal") != r["rule"].get("ideal") or c["rule"].get("bound") != r["rule"].get("bound"):
                messages.append(f"{name}: ideal values or bound differ")
            else:
                messages.extend(f"{name}: {f}" for f in bounded_failures(c))
        elif compare == "property":
            messages.extend(f"{name}: {f}" for f in property_failures(c))
    return ("fail" if messages else "pass"), messages


def check_run(run):
    problems = []
    for (identifier, variant), entry in run.entries.items():
        where = f"{variant}/{identifier}"
        document = run.case(identifier, variant)
        if document is None:
            problems.append(f"{where}: file missing")
            continue
        if run.digest(identifier, variant) != entry["sha256"]:
            problems.append(f"{where}: digest differs from run.json")
        check_case(document, problems, where)
    for problem in problems:
        print(problem)
    print(f"checked {len(run.entries)} case files: {len(problems)} problems")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.check:
        if len(args.runs) != 1:
            parser.error("--check takes one run directory")
        return check_run(Run(args.runs[0]))
    if len(args.runs) != 2:
        parser.error("give a reference and a candidate run directory")
    ref, cand = Run(args.runs[0]), Run(args.runs[1])
    same_profile = ref.device_key == cand.device_key
    prefixes = [p for p in args.only.split(",") if p]
    counts = {"pass": 0, "fail": 0, "incomparable": 0, "missing": 0}
    notes = []
    for (identifier, variant) in sorted(ref.entries):
        if prefixes and not any(identifier.startswith(p) for p in prefixes):
            continue
        r = ref.case(identifier, variant)
        c = cand.case(identifier, variant)
        if c is None:
            counts["missing"] += 1
            print(f"missing       {variant:11} {identifier}")
            continue
        verdict, messages = compare_case(r, c, same_profile, notes)
        counts[verdict] += 1
        if verdict != "pass" or args.verbose:
            print(f"{verdict:13} {variant:11} {identifier}")
            for message in messages:
                print(f"    {message}")
    for note in notes:
        print(f"note: {note}")
    profile = "same device profile" if same_profile else "different device profiles: device rules not compared"
    print(f"{ref.device_key} -> {cand.device_key} ({profile}): "
          + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts["fail"] or counts["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
