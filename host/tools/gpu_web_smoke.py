#!/usr/bin/env python3
"""Run the WebGPU conformance smoke test (webmetal_host::gpu::smoke) in a headless browser.

Builds the gpu_web_smoke example for wasm32 with the `web` feature, runs wasm-bindgen on it, serves the result
from 127.0.0.1 (a secure context, which WebGPU requires) and opens it in a headless browser. The page runs the
smoke test through the browser's WebGPU and posts its report back. The report's check lines are then compared with
host/tests/gpu_smoke_expected.txt and, with --compare, with another report (a native run kept with
WEBMETAL_SMOKE_REPORT).

Usage: gpu_web_smoke.py [--browser PATH] [--browser-arg ARG ...] [--cargo CARGO] [--out REPORT]
                        [--compare REPORT] [--timeout SECONDS] [--keep DIR]

The wasm-bindgen CLI on PATH (or --wasm-bindgen) must be the version host/Cargo.toml pins. Exit status 0 when the
report conforms to the expectations, 1 when it does not, 2 when the run could not complete.
"""

import argparse
import http.server
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.dirname(HERE)
ROOT = os.path.dirname(HOST)
EXPECTED = os.path.join(HOST, "tests", "gpu_smoke_expected.txt")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>WebMetal smoke</title>
<script type="module">
import init, { run_smoke } from "./gpu_web_smoke.js";
async function post(kind, text) {
  await fetch("/" + kind, { method: "POST", body: text });
}
try {
  await init();
  await post("report", await run_smoke());
} catch (e) {
  await post("failure", String(e && e.stack ? e.stack : e));
}
</script>
"""


def pinned_wasm_bindgen():
    with open(os.path.join(HOST, "Cargo.toml"), encoding="utf-8") as f:
        m = re.search(r'wasm-bindgen = \{ version = "=([0-9.]+)"', f.read())
    if not m:
        raise SystemExit("host/Cargo.toml does not pin wasm-bindgen")
    return m.group(1)


def checks(text):
    out = []
    for line in text.splitlines():
        if line.startswith("check ") and " = " in line:
            k, v = line[len("check "):].split(" = ", 1)
            out.append((k, v))
    return out


def conforms(expected, report):
    """The same rule as smoke::conforms: expected values may list alternatives separated by ' || '."""
    got = dict(checks(report))
    diff = []
    exp = checks(expected)
    for k, v in exp:
        g = got.get(k)
        if g is None:
            diff.append("%s: %s | <absent>" % (k, v))
        elif g not in v.split(" || "):
            diff.append("%s: %s | %s" % (k, v, g))
    keys = {k for k, _ in exp}
    for k, g in checks(report):
        if k not in keys:
            diff.append("%s: <not expected> | %s" % (k, g))
    return diff


def compare(a, b):
    ca, cb = dict(checks(a)), dict(checks(b))
    keys = [k for k, _ in checks(a)] + [k for k, _ in checks(b) if k not in ca]
    return ["%s: %s | %s" % (k, ca.get(k, "<absent>"), cb.get(k, "<absent>")) for k in keys if ca.get(k) != cb.get(k)]


def build(args, out_dir):
    env = dict(os.environ)
    cmd = [args.cargo, "build", "--target", "wasm32-unknown-unknown", "--features", "web", "--example", "gpu_web_smoke"]
    subprocess.run(cmd, cwd=HOST, env=env, check=True)
    target = env.get("CARGO_TARGET_DIR") or os.path.join(ROOT, "target")
    wasm = os.path.join(target, "wasm32-unknown-unknown", "debug", "examples", "gpu_web_smoke.wasm")
    want = pinned_wasm_bindgen()
    have = subprocess.run([args.wasm_bindgen, "--version"], capture_output=True, text=True, check=True).stdout.split()
    if len(have) < 2 or have[1] != want:
        raise SystemExit("%s is %s; host/Cargo.toml pins wasm-bindgen %s" % (args.wasm_bindgen, " ".join(have), want))
    subprocess.run([args.wasm_bindgen, "--target", "web", "--no-typescript", "--out-dir", out_dir, wasm], check=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(PAGE)


def serve(directory, results, done):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n).decode("utf-8", "replace")
            results[self.path.strip("/")] = body
            self.send_response(204)
            self.end_headers()
            done.set()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--browser", default=CHROME)
    ap.add_argument("--browser-arg", action="append", default=[])
    ap.add_argument("--cargo", default="cargo")
    ap.add_argument("--wasm-bindgen", default="wasm-bindgen")
    ap.add_argument("--out", help="write the report here")
    ap.add_argument("--compare", help="a report of another implementation to compare with")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--keep", help="build the page in this directory and keep it")
    args = ap.parse_args()

    page_dir = args.keep or tempfile.mkdtemp(prefix="webmetal-smoke-")
    os.makedirs(page_dir, exist_ok=True)
    build(args, page_dir)
    results, done = {}, threading.Event()
    httpd = serve(page_dir, results, done)
    url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    profile = tempfile.mkdtemp(prefix="webmetal-smoke-profile-")
    cmd = [args.browser, "--headless=new", "--no-first-run", "--no-default-browser-check",
           "--user-data-dir=" + profile] + args.browser_arg + [url]
    browser = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        finished = done.wait(args.timeout)
    finally:
        browser.terminate()
        try:
            browser.wait(10)
        except subprocess.TimeoutExpired:
            browser.kill()
        httpd.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
        if not args.keep:
            shutil.rmtree(page_dir, ignore_errors=True)
    if not finished:
        print("the page posted nothing within %.0f s" % args.timeout, file=sys.stderr)
        return 2
    if "failure" in results:
        print("the page failed:\n" + results["failure"], file=sys.stderr)
        return 2
    report = results["report"]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    impl = next((l for l in report.splitlines() if l.startswith("info implementation = ")), "")
    print(impl)
    with open(EXPECTED, encoding="utf-8") as f:
        diff = conforms(f.read(), report)
    print("against the specification: %s" % ("conforms" if not diff else "%d differences" % len(diff)))
    for d in diff:
        print("  " + d)
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            other = f.read()
        cd = compare(report, other)
        print("against %s: %s" % (args.compare, "identical check lines" if not cd else "%d differences" % len(cd)))
        for d in cd:
            print("  " + d)
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
