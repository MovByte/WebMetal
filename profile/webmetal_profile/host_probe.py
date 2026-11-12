"""Method B1: the host's own Metal.framework answering the same queries for the A10X driver.

`probe/metal_profile_probe.m` is compiled with the host SDK and given the driver answers that
method A extracted from the target's driver classes. It reports what this Mac's
`_MTLDevice` implementation computes from them. The host framework is a different build of
the same framework code (macOS, not iOS 17.7), executed natively, so agreement checks both
the extraction (emulation, decoding) and the framework tables it ran through.
"""

import json
import subprocess
from pathlib import Path

PROBE_SOURCE = Path(__file__).resolve().parents[1] / 'probe' / 'metal_profile_probe.m'


class ProbeError(Exception):
    pass


def build(build_dir):
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / 'metal_profile_probe'
    result = subprocess.run(['xcrun', 'clang', '-fobjc-arc', '-O1', '-Wall', '-Werror',
                             '-framework', 'Foundation', '-framework', 'Metal',
                             str(PROBE_SOURCE), '-o', str(binary)],
                            capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise ProbeError(f'probe build failed:\n{result.stderr}')
    return binary


def run(binary, request, build_dir):
    path = Path(build_dir) / 'metal_profile_probe_input.json'
    path.write_text(json.dumps(request, sort_keys=True, indent=1))
    result = subprocess.run([str(binary), str(path)], capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise ProbeError(f'probe exited with {result.returncode}:\n{result.stderr[-4000:]}')
    return json.loads(result.stdout)


def host_identity():
    product = subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True).stdout.strip()
    build = subprocess.run(['sw_vers', '-buildVersion'], capture_output=True, text=True).stdout.strip()
    return product, build
