# WebMetal wire protocol

The records the guest Metal plugin (`WebMetal.bundle`) and the host executor exchange, and the accelerator service
that carries them.

| Path | What it is |
| --- | --- |
| `schema/webmetal.toml` | The protocol: every record, structure and object kind, and the guest-local annotations. The only source of truth. |
| `gen/wiregen.py` | Generates everything below from the schema. `--check` fails if a committed output is stale. |
| `generated/WebMetalWire.h` | C++17 / Objective-C++ header for the guest: encoders for requests and stream records, decoders for replies and events, Foundation conversions. |
| `generated/WebMetalWireFuzz.h` | C++ random-record generators and comparisons used by the cross-language tests. |
| `generated/opcodes.md` | The opcode table. |
| `src/` | The `webmetal-wire` crate: `generated.rs` (generated) over a hand-written codec, property-list value and random-value runtime. |
| `tools/target_surface.py` | Derives the Metal driver surface of iPadOS 17.7.11 (21H461) and checks the schema against it. |
| `target/quartzcore-driver-selectors.tsv` | The QuartzCore driver-selector list that tool derives. |
| `tests/` | Round trips, golden encodings, malformed input, schema hash, and the C++/Rust cross-language round trip. |

## Records

All integers are little-endian. A record is an 8-byte header followed by the record's fields:

| Offset | Field | |
| --- | --- | --- |
| 0 | `opcode: u16` | Names the record. A reply's opcode is its request's opcode with bit 15 (`0x8000`) set. |
| 2 | `flags: u16` | Zero. |
| 4 | `length: u32` | Bytes in the whole record, header included; a multiple of 8. |

Fields follow in schema order. Each starts at the next offset, counted from the start of the record, that is a
multiple of its alignment; the record ends at the next multiple of 8 after its last field. Every skipped byte is zero.

| Schema type | Encoding | Alignment |
| --- | --- | --- |
| `u8`, `bool` | one byte; `bool` is 0 or 1 | 1 |
| `u16` | | 2 |
| `u32`, `i32`, `f32` | floats as their IEEE-754 bits | 4 |
| `u64`, `i64`, `f64` | | 8 |
| object kind `Kind` / `Kind?` | `u64` handle; 0 (nil) only for `Kind?` | 8 |
| `guest_addr` | `u64` virtual address in the task that sent the record | 8 |
| `surface` | `u32` IOSurfaceID | 4 |
| `block`, `out`, `present` | one byte: 1 if the caller passed a non-nil block, out-pointer or object that has no wire form | 1 |
| `Struct` | its fields, each aligned as above | per field |
| `Struct?` | one presence byte (0 or 1), then the struct if 1 | 1 |
| `str` | `u32` byte count, then UTF-8 bytes | 4 |
| `str?`, `url?` | as `str`, or the count `0xffffffff` alone for nil; `url?` is an NSURL's absolute string | 4 |
| `cstr?` | as `str?`, bytes without NUL and not necessarily UTF-8 | 4 |
| `bytes` | `u32` byte count, then the bytes | 4 |
| `[T]` | `u32` element count, then the elements | 4 |
| `[u8; N]` | N bytes | 1 |
| `value?` | a property-list value, see `src/value.rs` (tag byte, then payload; dictionary keys strictly ascending) | 1 |

A field written `name: type = expr` must have exactly as many elements as the value of `expr` (another field of the
record, or one member of a struct field), as when Metal passes a C array with its count or `NSRange`. Metal arguments
are carried verbatim, in Metal's order, even where the executor does not act on them yet.

The encoding is canonical. Decoders refuse a record with non-zero flags or padding, a length that is not exactly the
padded end of its fields, a bool or presence byte other than 0 or 1, a count of `0xffffffff` where nil is not
allowed, a count larger than the bytes left, a nil handle where Metal requires an object, invalid UTF-8, a broken
length constraint, unsorted or repeated dictionary keys, values nested deeper than 32, or a `Failure` that breaks its
rules. Anything that decodes re-encodes to the same bytes; the decoders never interpret malformed input.

### Handles

Object handles are 64-bit values the guest allocates, unique within a connection and never reused; 0 is nil. The same
value is the 64-bit word an argument buffer holds for a texture, sampler or buffer reference, so the shader compiler
and the executor decode argument buffers by this definition.

### Channels

**Control.** A request record goes to the host through the user client's `control` selector; the host answers with one
reply record. The reply's payload starts with `status: u32`. Status 0 is followed by the request's reply fields;
any other status is followed by the rest of a `Failure` (`domain: str?`, `code: i64`, `message: str`):

| Status | Meaning |
| --- | --- |
| 0 | Success. |
| 1 | A failure Metal defines: `domain` and `code` are the NSError Metal returns (or the nil result it implies). |
| 2 | Not implemented by the host: the guest reports it through the not-implemented channel and fails the call. |
| 3 | The host cannot honor a capability the device advertises (a WebGPU feature or limit); `message` names it. Reported like 2. |
| 4 | The request is malformed, names an unknown handle, or breaks the protocol. |

Only status 1 has a domain; statuses 2 to 4 have code 0.

**Stream.** A command buffer's records, written by the guest's encoders into its own memory without a kernel entry
per call. A render, compute or blit encoder's records follow its `Begin*` record and end with `EncoderEndEncoding`.
A parallel render encoder's sub-encoders are spliced, in the order they were created, between its
`BeginParallelRenderPass` and its own `EncoderEndEncoding`. `Commit` hands the host the stream's address and length,
the command buffer's state, and every handle the stream references; the host has consumed the stream before it
replies. An opcode the host does not implement still decodes: the host completes the command buffer with status 2 and
the record's offset, never skipping it.

**Events.** Host-to-guest records (command buffer scheduled and completed, device lost), queued per command queue.
The host rings the queue's doorbell; the guest collects the records with `FetchEvents`.

### Version

The protocol version is the SHA-256 of `schema/webmetal.toml` (`SCHEMA_HASH`, `kSchemaHash`). `Open`, the first
request on a connection, carries the guest's hash; the host refuses the connection (status 4) when it differs.

## Transport

The accelerator service takes the place of the GPU's own accelerator. Metal.framework matches services of category
`IOAcceleratorES`, reads `MetalPluginName` and `MetalPluginClassName` from the service, and loads that bundle from
`/System/Library/Extensions`, as Metal.framework 343.21 does on 21H461.

| Item | Value |
| --- | --- |
| Service class | `WebMetalAccelerator` |
| `IOMatchCategory` | `IOAcceleratorES` |
| `MetalPluginName` | `WebMetal` |
| `MetalPluginClassName` | `WebMetalDevice` |
| User client type | 0 |

User client selectors, all taking and returning 64-bit scalars:

| Selector | Inputs | Outputs | |
| --- | --- | --- | --- |
| 0 `control` | request address, request length, reply address, reply capacity | reply length | Runs one request read from the caller's memory and writes its reply. If the reply is longer than the capacity nothing is written and the reply stays pending. |
| 1 `fetch_reply` | reply address, reply capacity | reply length | Writes the pending reply of the last `control` call without running the request again. |
| 2 `register_event_port` | queue handle (0: device-level events) | | Asynchronous method: the host sends an async completion with the queue handle as its argument whenever events for that queue become pending. |

## Checking the schema against 21H461

```sh
python3 -B -I wire/tools/target_surface.py --root <21H461 root file system> check
python3 -B -I wire/tools/target_surface.py --root <21H461 root file system> derive --write
```

The tool reads the target's dyld shared cache read-only, decodes the Objective-C metadata of Metal.framework,
IOGPU.framework, the AGX driver bundle and QuartzCore, and checks:

- every op's arguments against the 21H461 extended method type encoding of each Metal method it carries;
- that every Metal method an op carries is implemented by the driver layers (the AGX bundle or IOGPU.framework, which
  the plugin replaces), not only by Metal.framework;
- every descriptor structure against the settable state of its 21H461 class (getter/setter pairs and readonly
  collection properties); a field may be left out only by an `excluded` entry that gives the reason;
- every guest-local annotation, verifying the reasons the binaries can prove: `framework` (only Metal.framework
  implements it), `absent` (an optional method no driver layer implements), `unavailable` (no driver layer implements
  it; API_UNAVAILABLE on iOS), `unreachable` (the factory that would hand out the object is itself absent);
- that `target/quartzcore-driver-selectors.tsv` equals a fresh derivation;
- that every QuartzCore driver selector declared by an `MTL*CommandEncoder`, `MTLDevice` or `MTLDeviceSPI` protocol has
  an opcode or a guest-local annotation.

It prints the remaining backlog: driver-protocol methods that have neither an opcode nor a guest-local annotation
yet. Those are not served; the guest plugin's generated stubs report them as not implemented.

The QuartzCore list is QuartzCore's `__objc_selrefs` intersected with the selectors the driver implements or the
`MTL*` protocols declare, without runtime-generated and root-class selectors; see the tool's docstring for the exact
definition. A selector reference shows that QuartzCore can send the message, not to which receiver.

## Generating and testing

```sh
python3 -B -I wire/gen/wiregen.py          # regenerate after editing the schema
cargo test -p webmetal-wire                 # needs python3 and a C++ compiler ($CXX, default xcrun clang++ on macOS)
```

The cross-language test builds `tests/cpp/crosslang.cpp` against the generated headers as C++17 and, on macOS, as
Objective-C++. From the same seeds, the C++ encoder and the Rust encoder must produce identical bytes for random
requests and stream records covering every opcode, the Rust decoder must reproduce them, and the C++ decoder must
reproduce random replies and events the Rust encoder wrote.
