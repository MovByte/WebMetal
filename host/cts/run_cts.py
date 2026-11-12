#!/usr/bin/env python3
"""Run a subset of the WebGPU CTS (https://github.com/gpuweb/cts) against a native WebGPU implementation.

Implementations, each a JavaScript host for the CTS command-line runner around the native library:

  dawn  Dawn's Node.js binding (dawn.node and its cts.cjs, from a Dawn build with DAWN_BUILD_NODE_BINDINGS=ON),
        run by Node.js with the CTS's tools/run_node. The CTS checkout needs `npm ci`.
  wgpu  wgpu's cts_runner (the Deno runtime with deno_webgpu over wgpu-core, built from the wgpu repository at the
        release the wgpu-native library is built from), run with the CTS's tools/run_deno from the checkout's built
        out/ directory (the gh-pages revisions carry it).

Each query of the list is run in its own runner process; every test case's status (pass, fail, warn, skip) is
recorded, and a query whose runner exits without reporting a case is recorded as `crash`. With --expect, the run is
compared with an earlier results file and the differences are printed; the exit status is then 1 if any case
changed, else 0.

Usage:
  run_cts.py --impl dawn --cts DIR --dawn-node DIR [--node NODE] [--list FILE] [--out FILE] [--expect FILE]
  run_cts.py --impl wgpu --cts DIR --cts-runner PATH [--list FILE] [--out FILE] [--expect FILE]
"""

import argparse
import datetime
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIST = os.path.join(HERE, "subset.lst")
RESULT_RE = re.compile(r"^\[(pass|fail|warn|skip)\] (\S+) \(([0-9.]+)ms\)\. Log:")


def queries(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def git_rev(path):
    r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def command(args, query):
    if args.impl == "dawn":
        provider = os.path.join(os.path.abspath(args.dawn_node), "cts.cjs")
        if not os.path.isfile(provider) or not os.path.isfile(os.path.join(args.dawn_node, "dawn.node")):
            raise SystemExit("%s must hold dawn.node and cts.cjs" % args.dawn_node)
        return [args.node, "tools/run_node", "--gpu-provider", provider, "--verbose", query]
    runner = os.path.abspath(args.cts_runner)
    if not os.path.isfile(os.path.join(args.cts, "out", "common", "runtime", "cmdline.js")):
        raise SystemExit("%s has no out/common/runtime/cmdline.js (use a revision of the gh-pages branch)" % args.cts)
    return [runner, "./tools/run_deno", "--verbose", query]


def run_query(args, query):
    """[(status, case)] for one query, and the runner's exit status and output tail."""
    cmd = command(args, query)
    try:
        p = subprocess.run(cmd, cwd=args.cts, capture_output=True, text=True, timeout=args.timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc = "timeout"
    results = []
    seen = set()
    for line in out.splitlines():
        m = RESULT_RE.match(line)
        if m and m.group(2) not in seen:
            # The summary repeats skipped, warned and failed cases; the first line of each case is its result.
            seen.add(m.group(2))
            results.append((m.group(1), m.group(2)))
    return results, rc, "\n".join(out.splitlines()[-15:])


def machine():
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        ver = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip()
        build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
        return "%s, macOS %s (%s)" % (chip, ver, build)
    return platform.platform()


def parse_results(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            status, case = line.rstrip("\n").split(" ", 1)
            out[case] = status
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["dawn", "wgpu"], required=True)
    ap.add_argument("--cts", required=True, help="a checkout of gpuweb/cts")
    ap.add_argument("--dawn-node", help="the Dawn build directory holding dawn.node and cts.cjs")
    ap.add_argument("--node", default="node")
    ap.add_argument("--cts-runner", help="wgpu's cts_runner executable")
    ap.add_argument("--describe", default="", help="the implementation's name and version, for the results header")
    ap.add_argument("--list", default=DEFAULT_LIST)
    ap.add_argument("--out", help="write the results here")
    ap.add_argument("--expect", help="compare with an earlier results file")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per query")
    args = ap.parse_args()
    if args.impl == "dawn" and not args.dawn_node:
        ap.error("--impl dawn needs --dawn-node")
    if args.impl == "wgpu" and not args.cts_runner:
        ap.error("--impl wgpu needs --cts-runner")

    qs = queries(args.list)
    rows = []
    counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0, "crash": 0}
    for q in qs:
        results, rc, tail = run_query(args, q)
        if not results:
            rows.append(("crash", q))
            counts["crash"] += 1
            print("crash   %s (exit %s)\n%s" % (q, rc, tail), file=sys.stderr)
            continue
        for status, case in results:
            rows.append((status, case))
            counts[status] += 1
        n = {s: sum(1 for st, _ in results if st == s) for s in ("pass", "fail", "warn", "skip")}
        print("%-7s %s: %d pass, %d fail, %d warn, %d skip" % ("done", q, n["pass"], n["fail"], n["warn"], n["skip"]))

    header = [
        "# WebGPU CTS subset results, written by host/cts/run_cts.py",
        "# implementation: %s (%s)" % (args.describe or args.impl, "dawn.node" if args.impl == "dawn" else "wgpu cts_runner"),
        "# cts: gpuweb/cts %s" % git_rev(args.cts),
        "# list: %s" % os.path.relpath(os.path.abspath(args.list), os.path.dirname(os.path.dirname(HERE))),
        "# machine: %s" % machine(),
        "# date: %s" % datetime.date.today().isoformat(),
        "# summary: %s" % ", ".join("%s %d" % (k, v) for k, v in counts.items()),
    ]
    text = "\n".join(header) + "\n" + "".join("%s %s\n" % (s, c) for s, c in rows)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(header[-1][2:])
    if args.expect:
        old = parse_results(args.expect)
        new = {c: s for s, c in rows}
        changed = sorted(set(old) | set(new))
        diffs = ["%s: %s -> %s" % (c, old.get(c, "absent"), new.get(c, "absent")) for c in changed if old.get(c) != new.get(c)]
        print("against %s: %s" % (args.expect, "no change" if not diffs else "%d changed" % len(diffs)))
        for d in diffs:
            print("  " + d)
        return 1 if diffs else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
