#!/usr/bin/env python3
"""Run the presentation sink on a page's canvas (host/embed-ref/examples/present_web.rs) in a headless browser.

Builds the present_web example for wasm32 with the `web` feature, runs wasm-bindgen on it, serves the result from
127.0.0.1 (a secure context, which WebGPU requires) and opens it in a headless browser with the real GPU. The page
presents IOSurface frames on canvases and reads back what the browser shows; it posts its report, whose check lines
say "exact" when every red, green and blue code equals the surface's and alpha is 255. Observe lines are measurements
the exit status does not depend on (see the example's documentation).

Usage: present_web.py [--browser PATH] [--browser-arg ARG ...] [--cargo CARGO] [--wasm-bindgen PATH] [--out REPORT]
                      [--timeout SECONDS] [--keep DIR]

--browser is Google Chrome by default; another Chromium-based browser or Firefox (by its executable name) may be
named instead. Each runs headless with a fresh profile. The wasm-bindgen CLI must be the version host/Cargo.toml pins.
Exit status 0 when every check is exact, 1 when one is not, 2 when the run could not complete.
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

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
HOST = os.path.dirname(CRATE)
ROOT = os.path.dirname(HOST)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>WebMetal presentation</title>
<body>
<script type="module">
import init, { run_present } from "./present_web.js";
async function post(kind, text) {
  await fetch("/" + kind, { method: "POST", body: text });
}
try {
  await init();
  await post("report", await run_present());
} catch (e) {
  await post("failure", String(e && e.stack ? e.stack : e));
}
</script>
</body>
"""


def pinned_wasm_bindgen():
    with open(os.path.join(HOST, "Cargo.toml"), encoding="utf-8") as f:
        m = re.search(r'wasm-bindgen = \{ version = "=([0-9.]+)"', f.read())
    if not m:
        raise SystemExit("host/Cargo.toml does not pin wasm-bindgen")
    return m.group(1)


def build(args, out_dir):
    cmd = [args.cargo, "build", "--target", "wasm32-unknown-unknown", "-p", "webmetal-embed-ref", "--features", "web",
           "--example", "present_web"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    target = os.environ.get("CARGO_TARGET_DIR") or os.path.join(ROOT, "target")
    wasm = os.path.join(target, "wasm32-unknown-unknown", "debug", "examples", "present_web.wasm")
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
            results[self.path.strip("/")] = self.rfile.read(n).decode("utf-8", "replace")
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
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--keep", help="build the page in this directory and keep it")
    args = ap.parse_args()

    version = subprocess.run([args.browser, "--version"], capture_output=True, text=True).stdout.strip()
    page_dir = args.keep or tempfile.mkdtemp(prefix="webmetal-present-")
    os.makedirs(page_dir, exist_ok=True)
    build(args, page_dir)
    results, done = {}, threading.Event()
    httpd = serve(page_dir, results, done)
    url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    profile = tempfile.mkdtemp(prefix="webmetal-present-profile-")
    if "firefox" in os.path.basename(args.browser).lower():
        cmd = [args.browser, "--headless", "--no-remote", "--profile", profile] + args.browser_arg + [url]
    else:
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
    report = "info browser = %s\n" % version + results["report"]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    print(report, end="")
    checks = [l for l in report.splitlines() if l.startswith("check ")]
    bad = [l for l in checks if not l.endswith(" = exact")]
    if not checks:
        print("the report has no checks", file=sys.stderr)
        return 2
    print("%d checks, %d exact" % (len(checks), len(checks) - len(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
