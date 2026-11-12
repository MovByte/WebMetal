# Reference output format

Format version: `webmetal-reference/1`.

`wmref run` writes one directory per run. The executor tests, the trace tools
and `compare.py` read it. All files are UTF-8 JSON. Object keys are sorted.
Floating-point values never appear as JSON numbers: they are written as IEEE
bit patterns in hexadecimal, so every value round-trips exactly.

## Run directory

| Path | Content |
| --- | --- |
| `run.json` | Run manifest: build, device, host, variants, case list |
| `release/<case>.json` | Case file for the `release` variant |
| `debug-layer/<case>.json` | Case file for the `debug-layer` variant |

The committed reference outputs live in `goldens/<device key>/` with the same
layout.

## Variants

Each case runs once per variant, in its own process. The runner removes every
`MTL_*` and `METAL_*` variable from the child's environment and then adds the
variant's variables.

| Variant | Variables | Purpose |
| --- | --- | --- |
| `release` | none | Metal without API validation |
| `debug-layer` | `MTL_DEBUG_LAYER=1` | Metal API validation in its default (assert) mode |

## run.json

| Key | Content |
| --- | --- |
| `format` | `webmetal-reference/1` |
| `build` | Harness build: `platform`, `sdk`, `sdk_version`, `sdk_build`, `clang`, `metal` (compiler version lines), `target`, `air_target`, `metal_std`, `source_sha256` (digest of the harness sources the binary was built from) |
| `device` | `name`, `architecture` (when the OS reports it), `families` (answer of `supportsFamily:` for each `MTLGPUFamily`), `key`, `source` (the `--device` specification) |
| `host` | `platform`, `os_version`, `os_build`, `kernel`, `model`, `metal_framework` (CFBundleVersion of Metal.framework) |
| `variants` | Variant name to the environment variables it adds |
| `cases` | One entry per case and variant: `case`, `variant`, `status`, `outcome` (`exit N`, `signal N` or `timeout`), `sha256` of the case file |

The device key is the device name, platform, OS version and OS build in lower
case, joined with `-` (for example `apple-m4-macos-27.0.1-26a434`). Two runs
have the same device profile when their keys are equal.

## Case file

| Key | Content |
| --- | --- |
| `format` | `webmetal-reference/1` |
| `case` | Case identifier, for example `blit.copy-buffer` |
| `area` | API area tag (see COVERAGE.md) |
| `title` | One-line description |
| `variant` | `release` or `debug-layer` |
| `status` | See below |
| `message` | Present for `unsupported` and `harness-error` |
| `outcome` | How the case process ended, see below |
| `records` | The case's records in program order, one per line |

| Status | Meaning |
| --- | --- |
| `ok` | The case ran to its end |
| `unsupported` | A requirement record has the value `false`; nothing after it ran |
| `terminated` | The process ended before the case finished (for example a Metal assertion) |
| `harness-error` | The harness could not run the case; a reference run never contains this status |

| Outcome `kind` | Fields |
| --- | --- |
| `exit` | `code`: the exit status |
| `signal` | `signal`: the terminating signal number |
| `timeout` | none; the runner killed the process after `--timeout` seconds |

Any outcome may carry `assertion`: the first Metal assertion printed by the
process, from the start of its line. A quoted validation message ends at its
closing quote (the first `'` followed by a line end or the end of the
output); any other assertion ends at its line end. A system log prefix
(date, time, process name and identifiers) is removed, and hexadecimal
numbers of 9 or more digits (object addresses) are replaced by `0x`. The
text depends on the OS build and is never compared.

## Records

Every record has `kind`. Output records (`value`, `data`, `error`, `call`)
also have `name`, unique within the case, and `rule`.

| Kind | Fields |
| --- | --- |
| `require` | `name`, `value` (boolean). A requirement evaluated through a documented query. |
| `step` | `name`. A progress marker placed before a call, so a terminated run shows the last call reached. |
| `value` | `type`, `value`, and `display` for floating-point types |
| `data` | `elem`, `count`, `hex`, optional `layout` |
| `error` | `nil` (boolean); when false also `domain`, `code` and `description` |
| `call` | `raised` (boolean); when true also `exception` (the Objective-C exception name) and `reason` |

Value types:

| `type` | `value` |
| --- | --- |
| `u64`, `i64` | JSON integer |
| `bool` | JSON boolean |
| `string` | JSON string or null |
| `strings` | JSON array of strings |
| `f32` | String `0x` followed by 8 hexadecimal digits (IEEE binary32 bits) |
| `f64` | String `0x` followed by 16 hexadecimal digits (IEEE binary64 bits) |

`data` records hold little-endian bytes as lowercase hexadecimal in `hex`.
`elem` is one of `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f16`,
`f32`, `f64`, and `count` is the number of elements.

`description` (error records) and `reason` (call records) are reference text.
They are never compared.

### Layouts

A `layout` object describes texture bytes: `format` (the `MTLPixelFormat`
name without its prefix), `width`, `height`, `depth`, `bytes_per_row`,
`bytes_per_image`. Cases may add keys that describe their own data; consumers
ignore keys they do not know.

## Rules

Each output record carries the rule that applies to it. The rule comes from
the Metal or Metal Shading Language definition of the behavior, never from a
tolerance chosen to make a comparison pass.

| `compare` | Meaning |
| --- | --- |
| `exact` | The comparison fields must be equal (see below). Used where Metal defines the result exactly. |
| `device` | A property of the device (a limit or a capability answer). Compared like `exact` between runs with the same device profile, and not compared otherwise. |
| `none` | Recorded for reference only. `reason` states why: the result is implementation-defined in Metal, it is text that depends on the OS build, or it is the result of a misuse. |
| `bounded` | Floating-point data compared with ideal values under an error model, see below. |
| `property` | The value must satisfy a named property, see below. |

Comparison fields by record kind: `value`: `type` and `value`; `data`:
`elem`, `count`, `hex` and `layout`; `error`: `nil`, `domain` and `code`;
`call`: `raised` and `exception`.

### Bounded comparisons

A bounded rule has `model`, `bound` (a decimal string), `reference` (the
clause that states the bound) and `ideal`: one IEEE binary64 value per data
element, as little-endian hexadecimal. Each measured element `v` with ideal
value `x` must satisfy:

| `model` | Condition |
| --- | --- |
| `ulp` | `abs(v - x) <= bound * ulp(x)`, where `ulp` is defined in MSL 4.1 section 8.4 for the record's element type (`f16` or `f32`): the gap between the two consecutive finite values around `x`, or, when `x` is representable, the distance from `x` to its nearest other finite value |
| `abs` | `abs(v - x) <= bound` |
| `srgb8-decode` | `x` is an 8-bit sRGB code; converting `v` back to an unrounded sRGB code with the rule of MSL 4.1 section 8.7.7 gives `r`, and `abs(r - x) <= bound` |

A NaN ideal value requires a NaN measured value.

The ideal values are computed by the case on the CPU from the definition of
the operation, so the reference run is itself checked against the stated
bound (`compare.py --check`).

### Properties

| `property` | Condition |
| --- | --- |
| `one-of` | The record's `value` is one of `values`. `reason` cites the definition that allows each of them (for example a status that may already have advanced when it is read). |

## Comparing two runs

`compare.py REFERENCE CANDIDATE` applies these steps to every case and
variant of the reference run:

1. The candidate must have the case file.
2. The sequence of requirement records must be equal. When it differs and the
   device profiles differ, the case is reported as not comparable; when the
   profiles are equal, it fails.
3. `status`, the outcome kind and its `code` or `signal` must be equal.
4. The sequence of step names must be equal.
5. Every output record of the reference must exist in the candidate with the
   same kind, type, rule and layout, and satisfy its rule. The candidate may
   not have output records the reference lacks.

`compare.py --check RUN` checks one run: every case file exists and matches
the digest in `run.json`, every record is well formed, `none` rules have a
reason, and every `bounded` and `property` record holds for the run's own
values.

## Generated inputs

Cases derive input bytes from `WMPattern(destination, length, seed)`: an
xorshift32 generator with state `seed` (or `0x9e3779b9` when `seed` is 0).
For each byte the state is updated with `s ^= s << 13; s ^= s >> 17;
s ^= s << 5` (32-bit unsigned arithmetic) and the byte is the low 8 bits of
the new state. The seeds are in the case sources.

## Versioning

A change that alters the meaning of an existing field changes the format
version. Adding record kinds, rule kinds, models or properties also changes
it, because older consumers cannot apply them.
