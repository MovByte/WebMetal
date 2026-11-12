# Metal reference harness

`wmref` is a set of small Objective-C programs, one Metal behavior each,
written against the public Metal API. Each program runs on the `MTLDevice` it
is given and records its outputs (buffer and texture bytes, values, errors,
status and handler sequences, reflection data) together with the comparison
rule that applies to each of them. Outputs recorded on real Metal are the
reference for the WebMetal executor tests and trace tools.

The same programs run unmodified on any device: the system's Metal device,
a device returned by a loaded library, or a device in another process
environment built for iOS.

## Layout

| Path | Content |
| --- | --- |
| `src/WMRef.h` | Entry points for running cases against a device |
| `src/WMRefCase.h` | Interface for writing cases |
| `src/Harness.m` | Case registry, record writer, helpers |
| `src/main.m` | `wmref` command-line runner |
| `src/cases/*.m` | The cases, grouped by API area |
| `shaders/*.metal` | Shaders used by the cases, compiled into `wmref.metallib` |
| `build.py` | Builds `wmref` and `wmref.metallib` |
| `compare.py` | Compares runs under the recorded rules |
| `FORMAT.md` | Output format and comparison rules |
| `COVERAGE.md` | API areas and the cases that cover them |
| `goldens/<device key>/` | Reference outputs recorded on real Metal |

## Requirements

| Tool | Version |
| --- | --- |
| macOS | 14 or newer (the harness itself targets macOS 14 and iOS 17) |
| Xcode | with the Metal toolchain (`xcrun metal`) |
| Python | 3.9 or newer |

## Build

```sh
cd tests/reference
python3 build.py                  # build/macos/wmref and build/macos/wmref.metallib
python3 build.py --platform ios   # build/ios: arm64 iOS 17.0 and newer
```

`--out DIR` writes the build elsewhere. Shaders are compiled with
`-std=macos-metal2.4` or `-std=ios-metal2.4`. `build.json` next to the binary
records the SDK, the compiler versions and a digest of the sources; the same
description is compiled into the binary and written into every run manifest.

## Run

```sh
build/macos/wmref list                          # identifier, area, title
build/macos/wmref info                          # device and host description
build/macos/wmref run --out RUN                 # every case, both variants
build/macos/wmref run --out RUN --only blit.,compute.atomics --variants release
build/macos/wmref case --case blit.copy-buffer  # one case in this process
```

`run` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--out DIR` | required | Run directory; must not exist or be empty |
| `--device SPEC` | `default` | Device to test, see below |
| `--library PATH` | `wmref.metallib` next to the binary | Shader library for the device's platform |
| `--variants LIST` | `release,debug-layer` | Variants to run (FORMAT.md, "Variants") |
| `--only PREFIXES` | all cases | Comma-separated case identifier prefixes |
| `--timeout SECONDS` | `120` | Per case and variant; the case process is killed after it |
| `--logs DIR` | none | Writes each case process's standard output and error |

`run` starts one process per case and variant (`wmref child`), so a Metal
assertion or a crash ends only that case and becomes its recorded outcome.

Device specifications:

| `--device` | Device |
| --- | --- |
| `default` | `MTLCreateSystemDefaultDevice()` |
| `name:NAME` | The device named `NAME` in `MTLCopyAllDevices()` (macOS) |
| `provider:PATH` | `dlopen(PATH)` and call `WMRefProvideDevice()` |
| `provider:PATH:SYMBOL` | `dlopen(PATH)` and call `SYMBOL()` |

A provider function has the type `id<MTLDevice> (*)(void)` and returns a
device that it keeps alive for the life of the process:

```objc
#import <Metal/Metal.h>

id<MTLDevice> WMRefProvideDevice(void) {
    static id<MTLDevice> device;
    if (!device) device = MTLCreateSystemDefaultDevice(); // or any other device
    return device;
}
```

To run cases inside another program, compile `src/*.m` (except `main.m`) and
`src/cases/*.m` into it and call `WMRefRunCase(device, libraryURL,
identifier, sink)` from `WMRef.h`. It runs one case in the calling process
and passes each record to `sink`.

## Devices

The cases target Apple-family GPUs. They use the features that the Metal
Feature Set Tables list for GPU family Apple3 and earlier without checking
for them, and record a requirement (`family.appleN`, or the device's own
query where Metal provides one) before using any feature listed for a later
family. `wmref run` stops, and `WMRefRunCase` returns `harness-error`, for a
device that does not report `supportsFamily:MTLGPUFamilyApple3`. The family
gates cite the Metal Feature Set Tables dated 2026-10-23.

## Compare

```sh
python3 compare.py goldens/apple-m4-macos-27.0.1-26a434 RUN   # reference against a run
python3 compare.py --check RUN                                # one run on its own
```

The exit status is 0 when every case passes. Cases whose requirements
differ between runs on different device profiles are reported as not
comparable, and `device` rules are compared only between runs with the same
device profile. FORMAT.md describes each step.

## Reference outputs

`goldens/<device key>/` holds a complete run of both variants on one real
device. Its `run.json` records the device name, GPU architecture and family
answers, the OS version and build, the Metal framework version, the SDK and
compiler versions, and the digest of the harness sources that produced it.

A reference run is accepted when:

1. `compare.py --check` reports no problems (no `harness-error`, every
   bounded and property record holds);
2. every case that is not a misuse case has status `ok` or `unsupported` in
   both variants and exits with status 0, and the `debug-layer` variant
   reports no assertion for it;
3. a second run on the same device compares equal to it and its case files
   are byte-identical.

Steps to record a new reference:

```sh
python3 build.py
build/macos/wmref run --out RUN1
build/macos/wmref run --out RUN2
python3 compare.py --check RUN1
python3 compare.py RUN1 RUN2
diff -r RUN1/release RUN2/release && diff -r RUN1/debug-layer RUN2/debug-layer
mkdir -p goldens/KEY && cp -R RUN1/run.json RUN1/release RUN1/debug-layer goldens/KEY/
```

`KEY` is `device.key` in `RUN1/run.json`. A change to the harness sources
requires recording the reference again, so that `build.source_sha256` in the
committed manifest matches the sources next to it.

## Writing a case

```objc
#import "WMRefCase.h"

WMREF_CASE(copyBuffer, "blit.copy-buffer", "blit.copy", "Copy a range between two buffers")
{
    uint8_t input[256];
    WMPattern(input, sizeof input, 1);
    id<MTLBuffer> a = [c bufferWithBytes:input length:sizeof input];
    id<MTLBuffer> b = [c buffer:256];
    [c blit:^(id<MTLBlitCommandEncoder> e) {
        [e copyFromBuffer:a sourceOffset:16 toBuffer:b destinationOffset:32 size:64];
    }];
    [c data:@"destination" elem:WMElemU8 bytes:b.contents length:b.length rule:WMExact()];
}
```

- One behavior per case. The identifier starts with its area group; the area
  tag is listed in COVERAGE.md.
- Check optional features with a documented query and record it with
  `[c require:value:]`. A false requirement ends the case as `unsupported`;
  never skip work silently.
- Give every output the rule its definition supports. `WMExact()` only where
  Metal or the Metal Shading Language defines the result exactly (cite the
  clause in a comment); `bounded:` with the clause that states the bound and
  ideal values computed on the CPU; `WMDevice()` for device properties;
  `WMNone(reason)` for implementation-defined results, OS-specific text and
  results of misuse. Never use a tolerance that no definition states.
- Record only values that are stable between runs. Reduce timestamps and
  other varying values to properties computed in the case (for example
  "start time is not after end time").
- Use `WMPattern` for input data and keep resources small.
- In misuse cases, place `[c step:]` before each call that may fail, so a
  terminated run shows which call ended it.
- Shader function names start with the prefix of their file.
