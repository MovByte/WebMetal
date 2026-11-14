#!/usr/bin/env python3
"""Run the executor's reference cases (the test files of host/tests that replay the Metal reference harness) in a
browser, on its WebGPU and the machine's GPU, and compare each with the reference outputs as the native runs do.

Builds the browser modules of host/web-tests (one per test file) for wasm32 with the `web` feature, runs wasm-bindgen
on them, builds the reference shader library for iOS (tests/reference/build.py, as the native pipeline tests do) and
serves the page, the modules and the repository's files from 127.0.0.1. The page runs every case in a fresh Worker
and posts each outcome back; a case passes when its test function returns, exactly as under `cargo test`.

Usage:
  web_corpus.py run [--browser chrome|chrome-headless|firefox|safari|PATH] [--modules a,b] [--case module.name] [--out RESULTS]
                    [--build DIR] [--no-build] [--reference-build DIR] [--timeout SECONDS] [--total SECONDS]
  web_corpus.py table RESULTS... [--native NAME=LOG ...]

Browsers:
  chrome   Google Chrome, in a window of a fresh profile (chrome-headless: headless, which Chrome 157 on the
           reference Mac does not keep running past about 30 s).
  firefox  Firefox, in a window of a fresh profile: Firefox gives a headless page no WebGPU adapter (measured on
           Firefox 151), and JavaScript Promise Integration, which the blocking test harness needs, is behind
           `javascript.options.wasm_js_promise_integration`, which the profile sets.
  safari   Safari, through `open -a Safari`: Safari has no headless mode and no profile of its own for a run.

The tests' own requirements hold here as natively: the case files of tests/reference/goldens, and Xcode's Metal
toolchain for the reference library (or --reference-build naming a build of `tests/reference/build.py --platform
ios`). The wasm-bindgen CLI on PATH must be the version host/Cargo.toml pins. Exit status: 0 when every case passed,
1 when one did not, 2 when the run could not complete.
"""

import argparse
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.dirname(HERE)
ROOT = os.path.dirname(HOST)
CRATE = os.path.join(HOST, "web-tests")
PAGE = os.path.join(CRATE, "page")
MODULES = ["blit_exec", "compute_exec", "mem_buffers", "mem_exec", "mem_textures", "pipeline_exec", "render_cases",
           "tess_exec", "view_exec"]
BROWSERS = {
    "chrome": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "chrome-headless": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "firefox": "/Applications/Firefox.app/Contents/MacOS/firefox",
    "safari": "/Applications/Safari.app",
}
FIREFOX_PREFS = {
    "javascript.options.wasm_js_promise_integration": "true",
    "browser.shell.checkDefaultBrowser": "false",
    "datareporting.policy.dataSubmissionEnabled": "false",
    "browser.aboutwelcome.enabled": "false",
}


def pinned_wasm_bindgen():
    with open(os.path.join(HOST, "Cargo.toml"), encoding="utf-8") as f:
        m = re.search(r'wasm-bindgen = \{ version = "=([0-9.]+)"', f.read())
    if not m:
        raise SystemExit("host/Cargo.toml does not pin wasm-bindgen")
    return m.group(1)


def build(args, out):
    env = dict(os.environ)
    cmd = [args.cargo, "build", "--target", "wasm32-unknown-unknown", "--features", "web", "--release", "--examples",
           "-p", "webmetal-web-tests"]
    subprocess.run(cmd, cwd=CRATE, env=env, check=True)
    target = env.get("CARGO_TARGET_DIR") or os.path.join(ROOT, "target")
    want = pinned_wasm_bindgen()
    have = subprocess.run([args.wasm_bindgen, "--version"], capture_output=True, text=True, check=True).stdout.split()
    if len(have) < 2 or have[1] != want:
        raise SystemExit("%s is %s; host/Cargo.toml pins wasm-bindgen %s" % (args.wasm_bindgen, " ".join(have), want))
    for name in MODULES:
        wasm = os.path.join(target, "wasm32-unknown-unknown", "release", "examples", name + ".wasm")
        dest = os.path.join(out, name)
        shutil.rmtree(dest, ignore_errors=True)
        subprocess.run([args.wasm_bindgen, "--target", "web", "--no-typescript", "--out-dir", dest, wasm], check=True)


def reference(args, out):
    if args.reference_build:
        return os.path.abspath(args.reference_build)
    dest = os.path.join(out, "reference-ios")
    if not os.path.isfile(os.path.join(dest, "wmref.metallib")):
        script = os.path.join(ROOT, "tests", "reference", "build.py")
        subprocess.run([sys.executable, "-B", "-I", script, "--platform", "ios", "--out", dest], check=True)
    return dest


class Run:
    def __init__(self, path):
        self.path = path
        self.environment = None
        self.results = []
        self.done = threading.Event()
        self.failure = None
        self.lock = threading.Lock()

    def save(self, browser):
        with self.lock:
            data = {"browser": browser, "environment": self.environment, "results": self.results,
                    "failure": self.failure, "complete": self.done.is_set()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)


def serve(build_dir, reference_dir, run, browser):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=ROOT, **kw)

        def translate_path(self, path):
            p = urllib.parse.unquote(urllib.parse.urlsplit(path).path)
            parts = [x for x in p.split("/") if x and x not in (".", "..")]
            if parts[:1] == ["__web"]:
                rest = parts[1:]
                if len(rest) == 1 and rest[0] in ("index.html", "worker.js"):
                    return os.path.join(PAGE, rest[0])
                return os.path.join(build_dir, *rest)
            if parts[:2] == ["__build", "reference-ios"]:
                return os.path.join(reference_dir, *parts[2:])
            return os.path.join(ROOT, *parts)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            p = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            if p.startswith("/__list/"):
                d = os.path.join(ROOT, *[x for x in p[len("/__list/"):].split("/") if x and x not in (".", "..")])
                if not os.path.isdir(d):
                    self.send_error(404)
                    return
                body = "\n".join(sorted(os.listdir(d))).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8", "replace") or "{}")
            with run.lock:
                if self.path == "/__environment":
                    run.environment = body
                elif self.path == "/__result":
                    run.results.append(body)
                    status = body.get("status")
                    name = "%s.%s" % (body.get("module"), body.get("case"))
                    print("%-60s %s" % (name, status), flush=True)
                elif self.path == "/__done":
                    run.failure = body.get("failure")
            if self.path == "/__done":
                run.done.set()
            run.save(browser)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def launch(browser, url, profile, log=None):
    path = BROWSERS.get(browser, browser)
    name = os.path.basename(path).lower()
    if browser == "safari" or name.startswith("safari"):
        subprocess.run(["open", "-g", "-a", path, url], check=True)
        return None
    if "firefox" in name:
        with open(os.path.join(profile, "user.js"), "w", encoding="utf-8") as f:
            for k, v in FIREFOX_PREFS.items():
                f.write('user_pref("%s", %s);\n' % (k, v))
        cmd = [path, "--no-remote", "--new-instance", "--profile", profile, url]
    else:
        # In a window: a headless Chrome 157 on the reference Mac receives SIGTERM about 30 s after it starts
        # ("Handling shutdown for signal 15" in its log; a windowed one runs on), which ends a run. The window is
        # kept running at full rate when it is in the background.
        cmd = [path, "--no-first-run", "--no-default-browser-check", "--user-data-dir=" + profile,
               "--disable-background-networking", "--disable-component-update", "--disable-background-timer-throttling",
               "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows", "--window-size=640,480",
               url]
        if browser == "chrome-headless":
            cmd.insert(1, "--headless=new")
    out = open(log, "w") if log else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT if log else subprocess.DEVNULL)


def close_safari_tabs(prefix):
    """Close the Safari tabs the run opened (`open -a` leaves them open)."""
    script = [
        'tell application "Safari"',
        "repeat with w in windows",
        'set ts to (every tab of w whose URL starts with "%s")' % prefix,
        "repeat with i from (count of ts) to 1 by -1",
        "close (item i of ts)",
        "end repeat",
        "end repeat",
        "end tell",
    ]
    args = ["osascript"]
    for line in script:
        args += ["-e", line]
    subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)


def run_main(args):
    out = args.build or tempfile.mkdtemp(prefix="webmetal-web-corpus-")
    os.makedirs(out, exist_ok=True)
    if not args.no_build:
        build(args, out)
    reference_dir = reference(args, out)
    results = args.out or os.path.join(out, "results-%s.json" % args.browser)
    run = Run(results)
    httpd = serve(out, reference_dir, run, args.browser)
    modules = args.modules or ",".join(MODULES)
    query = {"modules": modules, "timeout": str(args.timeout)}
    if args.case:
        query["case"] = args.case
    url = "http://127.0.0.1:%d/__web/index.html?%s" % (httpd.server_address[1], urllib.parse.urlencode(query))
    profile = tempfile.mkdtemp(prefix="webmetal-web-corpus-profile-")
    proc = launch(args.browser, url, profile, args.browser_log)
    print("%s: %s" % (args.browser, url), flush=True)
    try:
        finished = False
        deadline = time.monotonic() + args.total
        while not finished and time.monotonic() < deadline:
            finished = run.done.wait(5)
            if not finished and proc is not None and proc.poll() is not None:
                run.failure = "the browser exited (status %s) before the page finished" % proc.returncode
                break
    finally:
        if proc is None:
            close_safari_tabs("http://127.0.0.1:%d/" % httpd.server_address[1])
        else:
            proc.terminate()
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
        httpd.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
        run.save(args.browser)
    if run.failure:
        print("the page failed:\n" + run.failure, file=sys.stderr)
        return 2
    if not finished:
        print("the page did not finish within %.0f s" % args.total, file=sys.stderr)
        return 2
    if run.failure:
        print("the page failed:\n" + run.failure, file=sys.stderr)
        return 2
    passed = sum(1 for r in run.results if r.get("status") == "pass")
    print("%s: %d of %d cases passed; results in %s" % (args.browser, passed, len(run.results), results))
    return 0 if passed == len(run.results) else 1


def cell(r):
    if r is None:
        return "not run"
    s = r.get("status")
    if s == "pass":
        return "pass"
    why = (r.get("panic") or r.get("error") or "").splitlines()
    return "%s: %s" % (s, why[0][:160] if why else "")


def native_results(path):
    """Outcomes of `cargo test` output: `test NAME ... ok|FAILED|ignored`, keyed by test name."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"test (\S+) \.\.\. (ok|FAILED|ignored)", line)
            if m:
                out[m.group(1)] = {"status": "pass" if m.group(2) == "ok" else m.group(2).lower()}
    return out


def table_main(args):
    columns, rows = [], {}
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        env = data.get("environment") or {}
        title = data.get("browser") or path
        ua = env.get("userAgent", "")
        m = re.search(r"(Chrome|Firefox|Version)/([0-9.]+)", ua)
        if m:
            title += " " + m.group(2)
        columns.append(title)
        for r in data.get("results", []):
            key = "%s.%s" % (r.get("module"), r.get("case"))
            rows.setdefault(key, {})[title] = r
    for spec in args.native or []:
        name, path = spec.split("=", 1)
        columns.append(name)
        for test, r in native_results(path).items():
            for key in rows:
                if key.split(".", 1)[1] == test:
                    rows[key][name] = r
    print("| case | " + " | ".join(columns) + " |")
    print("| --- | " + " | ".join("---" for _ in columns) + " |")
    for key in sorted(rows):
        print("| %s | %s |" % (key, " | ".join(cell(rows[key].get(c)).replace("|", "\\|") for c in columns)))
    totals = []
    for c in columns:
        rs = [rows[k].get(c) for k in rows]
        totals.append("%d/%d" % (sum(1 for r in rs if r and r.get("status") == "pass"), sum(1 for r in rs if r)))
    print("| **passed** | " + " | ".join(totals) + " |")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--browser", default="chrome")
    r.add_argument("--modules", help="comma-separated module names (default: all)")
    r.add_argument("--case", help="run only MODULE.NAME")
    r.add_argument("--cargo", default="cargo")
    r.add_argument("--wasm-bindgen", default="wasm-bindgen")
    r.add_argument("--build", help="build into this directory and keep it")
    r.add_argument("--no-build", action="store_true", help="use the modules already in --build")
    r.add_argument("--reference-build", help="a build directory of tests/reference/build.py --platform ios")
    r.add_argument("--out", help="write the results here (JSON)")
    r.add_argument("--timeout", type=float, default=900, help="seconds a case may take")
    r.add_argument("--total", type=float, default=4 * 3600, help="seconds the whole run may take")
    r.add_argument("--browser-log", help="write the browser's own output here")
    t = sub.add_parser("table")
    t.add_argument("results", nargs="+")
    t.add_argument("--native", action="append", help="NAME=LOG: a `cargo test` log of a native run, as a column")
    args = ap.parse_args()
    return run_main(args) if args.cmd == "run" else table_main(args)


if __name__ == "__main__":
    sys.exit(main())
