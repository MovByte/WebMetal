"""Extraction of the A10X Metal device profile from an iPadOS 17.7.11 (21H461) root.

`build.collect` runs the primary method (`target`: the target's own Metal driver code under
emulation) and the cross-checks (`host_probe`: this Mac's Metal.framework; `static_decode`:
LLVM's disassembler; `sampler`: the host Metal compiler); `render.render` writes the TOML
profile. UniXNU's scripts/audit-target-metal-profile.py is the entry point.
"""
