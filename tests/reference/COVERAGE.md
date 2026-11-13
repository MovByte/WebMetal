# Coverage

This page lists the Metal behaviors the cases cover, grouped by API area,
and the behaviors that have no case yet with the reason. `wmref list` prints
every case with its area tag. A case whose records use `none` rules still
covers its behavior: the run records what real Metal does, and FORMAT.md
states that the value is not compared.

## Device, queues and command buffers

| Behavior | Cases |
| --- | --- |
| Device name, architecture, `supportsFamily:` and `supportsFeatureSet:` answers | `device.identity`, `device.families` |
| Capability getters and limits | `device.limits`, `device.texture-sample-counts`, `device.alignments`, `device.sparse-page-sizes`, `compute.pipeline-properties` |
| Command queue capacity (`maxCommandBufferCount`) | `queue.max-command-buffer-count` |
| Execution order within a queue | `command-buffer.enqueue-order` |
| Command buffer status | `command-buffer.status-sequence` |
| Scheduled and completed handlers, `waitUntilScheduled`, `waitUntilCompleted` | `command-buffer.handler-order`, `command-buffer.status-sequence` |
| `error`, encoder execution status, `logs` | `command-buffer.success-error-and-logs`, `command-buffer.purgeable-empty-resource` |
| GPU and kernel start and end times | `command-buffer.timing` |
| Retained and unretained references | `command-buffer.retained-references` |
| `MTLEvent`, `MTLSharedEvent`, listeners, shared event handles | `event.cross-queue-signal-wait`, `event.shared-cpu-signal`, `event.shared-listener`, `event.shared-handle` |
| `MTLFence` | `fence.across-encoders`, `fence.wait-before-later-update` |
| Hazard tracking modes | `hazard.tracking-mode-readback`, `fence.across-encoders` |
| Parallel render encoders | `pass.parallel-encoder-order` |
| Labels, debug groups, signposts | `debug.labels-and-groups` |
| Counter sets and timestamp sampling | `device.counter-sets`, `counters.timestamps` |
| `MTLIOCommandQueue` | `io.load-buffer` |
| Misuse | `command-buffer.commit-twice`, `command-buffer.commit-with-active-encoder`, `command-buffer.completed-handler-after-commit`, `command-buffer.scheduled-handler-after-commit`, `command-buffer.encoder-released-without-end-encoding`, `command-buffer.enqueue-after-commit`, `command-buffer.encoder-while-encoder-open`, `event.signal-with-open-encoder`, `fence.update-then-wait-same-encoder`, `io.load-past-end` |

## Storage, memory and coherence

| Behavior | Cases |
| --- | --- |
| Shared buffers | `storage.shared-roundtrip`, `storage.new-buffer-with-bytes-copies`, `storage.new-buffer-initial-contents` |
| CPU and GPU access to disjoint ranges between commits | `storage.coherence-interleaving` |
| `newBufferWithBytesNoCopy:` and its deallocator | `storage.no-copy-deallocator` |
| Private buffers and textures | `storage.private-buffer`, the `blit.` cases |
| Memoryless attachments | `storage.memoryless-depth`, `pass.memoryless-depth` |
| CPU cache mode, resource options, managed storage | `storage.resource-options`, `storage.write-combined`, `storage.managed-synchronize` |
| `replaceRegion:` and `getBytes:` | `storage.texture-replace-region-ordering`, `storage.texture-get-bytes-after-gpu-write`, `blit.optimize-contents` |
| Buffer-backed textures and texture buffers | `storage.buffer-backed-texture`, `texture.type-texture-buffer`, `texture.type-texture-buffer-device` |
| IOSurface-backed textures | `storage.iosurface-texture` |
| Heaps: automatic, placement, sizes and alignments, aliasing | `heap.automatic`, `heap.placement`, `heap.size-and-align`, `heap.make-aliasable`, `device.heap-size-align` |
| Purgeable state | `resource.purgeable-state` |
| Allocated sizes, working set, unified memory, residency set support | `resource.allocated-size`, `resource.device-memory` |
| Structure layout and access in device memory | `compute.memory-layout` |
| Misuse | `storage.buffer-length-zero`, `storage.memoryless-store-action`, `storage.private-buffer-contents`, `storage.private-texture-replace-region`, `storage.no-copy-unaligned`, `resource.make-aliasable-non-heap` |

## Textures and pixel formats

| Behavior | Cases |
| --- | --- |
| Texture types 1D, 1D array, 2D, 2D array, 3D, cube, cube array, 2D multisample, 2D multisample array, texture buffer, with out-of-bounds reads | `texture.type-1d`, `texture.type-1d-array`, `texture.type-2d`, `texture.type-2d-array`, `texture.type-3d`, `texture.type-cube`, `texture.cube-sample-axes`, `texture.type-cube-array`, `texture.type-2d-multisample`, `texture.type-2d-multisample-array`, `texture.type-texture-buffer`, `texture.type-texture-buffer-device` |
| Writes, including out-of-bounds writes and writes at every level and at levels out of range of textures and views | `texture.write-out-of-bounds`, `texture.write-levels`, `texture.fragment-access` |
| Views: format reinterpretation, sRGB, level and slice ranges, type changes, swizzle | `texture.view-format-reinterpret`, `texture.view-srgb-without-usage`, `texture.view-levels-slices`, `texture.view-swizzle` |
| Mipmap generation | `texture.generate-mipmaps` |
| Descriptor defaults and texture properties | `texture.properties` |
| Ordinary color formats (8, 16, 32 bits per channel; packed 32-bit; RG11B10; RGB9E5) | `format.<format>.upload-read`, `format.<format>.shader-write`, `format.<format>.render` |
| 16-bit normalized formats, A8Unorm, R8 and RG8 sRGB, packed 16-bit formats, BGR10A2 and extended-range formats | the same three cases per format |
| Three-channel formats, GBGR422, BGRG422 | `format.RGB*.texture-buffer-read`, `format.GBGR422.upload-read`, `format.BGRG422.upload-read` |
| PVRTC, EAC and ETC2, ASTC (LDR, sRGB, HDR), BC1 to BC7 | `format.<format>.blocks` |
| Depth and stencil formats, including 24-bit depth where the device reports it | `format.Depth16Unorm.upload-read`, `format.Depth32Float.upload-read`, `format.Stencil8.upload-read`, `format.Depth32Float_Stencil8.upload-read`, `format.Depth24Unorm_Stencil8.upload-read` |
| Misuse | `texture.error-1d-mipmapped`, `texture.error-too-many-levels`, `texture.error-multisample-mipmapped`, `texture.error-view-size-mismatch`, `texture.error-view-without-usage` |

## Samplers and depth-stencil state

| Behavior | Cases |
| --- | --- |
| Filters, clamp-to-edge, repeat, mirror repeat, per-axis modes | `sampler.address-modes`, `sampler.address-mode-r`, `sampler.filtering` |
| Clamp to zero, border colors, mirror clamp to edge | `sampler.address-modes`, `sampler.border-colors`, `sampler.address-mode-mirror-clamp-to-edge` |
| LOD clamps and selection, per-binding LOD clamps, `min_lod_clamp` | `sampler.lod-selection`, `sampler.binding-lod-clamps`, `sampler.min-lod-clamp` |
| Compare functions and gather | `sampler.compare-functions`, `sampler.gather` |
| Unnormalized coordinates | `sampler.unnormalized-coordinates` |
| Samplers declared in MSL | `sampler.constexpr` |
| Descriptor properties (`supportArgumentBuffers`, `lodAverage`, labels) | `sampler.properties` |
| Depth compare and write, stencil operations, masks, separate front and back references | `depth-stencil.compare-functions`, `depth-stencil.stencil-operations`, `depth-stencil.stencil-read-mask`, `depth-stencil.separate-front-back` |
| Depth bias and slope | `depth-stencil.depth-bias` |
| Depth clip mode | `raster.depth-clip-clamp` |
| Triangle fill mode | `raster.fill-mode-lines` |
| Viewport with depth range, scissor | `raster.viewport`, `raster.scissor` |
| Cull mode and front-facing winding | `raster.cull-winding` |
| Blend factors, operations, blend color, dual source, write masks | `blend.factors-operations`, `blend.dual-source`, `blend.write-mask` |
| Sample counts, coverage, sample mask, default positions, alpha to coverage | `msaa.coverage-4x`, `msaa.coverage-2x`, `msaa.default-sample-positions`, `msaa.alpha-to-coverage` |
| Visibility results (boolean and counting) | `pass.visibility-results` |
| Load and store actions, clear values, deferred store actions | `pass.load-store-actions`, `pass.clear-values`, `pass.store-action-deferred` |
| Color, depth and stencil resolve | `msaa.color-resolve`, `msaa.depth-resolve`, `msaa.depth-stencil-resolve` |
| Misuse | `sampler.error-unnormalized-repeat`, `sampler.error-unnormalized-mip-filter`, `sampler.error-unnormalized-different-filters`, `sampler.error-anisotropy-zero`, `sampler.error-lod-min-above-max`, `depth-stencil.error.test-without-attachment`, `raster.error.scissor-outside`, `msaa.error.sample-count-mismatch`, `pass.error.memoryless-store`, `pass.error.store-action-unknown-unset`, `pass.error.visibility-offset-unaligned`, `pass.error.visibility-offset-reused`, `blend.error.integer-target` |

## Render command encoder

| Behavior | Cases |
| --- | --- |
| Triangle coverage, edges, points, lines | `raster.triangle-coverage`, `raster.edge-ties`, `raster.points`, `raster.lines` |
| Primitive restart in strips | `draw.primitive-restart` |
| Indexed draws, index types and offsets | `draw.indexed` |
| Instancing, base vertex, base instance, vertex and instance ids | `draw.vertex-instance-ids`, `draw.indexed` |
| Indirect draws | `draw.indirect` |
| Vertex descriptors: formats, step functions, buffer offsets, dynamic stride | `vertex.formats`, `vertex.formats-packed-float`, `vertex.step-functions`, `vertex.step-base-instance`, `vertex.bytes-and-buffer-offset`, `vertex.attribute-stride` |
| `setVertexBytes:` | `vertex.bytes-and-buffer-offset` |
| Fragment discard, interpolation, depth output | `render.discard`, `render.interpolation`, `depth-stencil.fragment-depth` |
| Framebuffer fetch and raster order groups | `render.framebuffer-fetch`, `render.raster-order-groups` |
| Memory barriers between draws | `render.memory-barrier` |
| Tessellation | `pass.tessellation` |
| Tessellated points and triangle order, every partition mode over a factor sweep | `tessellation.quad`, `tessellation.triangle` |
| Tessellation factor culling, clamping, rounding, scale, maximum factor, winding | `tessellation.factors`, `tessellation.misuse` |
| Factor step functions, instance strides, patch start, patch index buffers, instance and patch order | `tessellation.draws`, `tessellation.misuse` |
| Rendered patch draws: control points and per-patch attributes, indexed control points, culled patches, triangle order | `tessellation.render` |
| Layered rendering | `pass.layered-rendering`, `texture.type-2d-multisample-array` |
| Pipelines with rasterization disabled | `draw.rasterization-disabled`, `pipeline.rasterization-disabled` |
| Misuse | `draw.error.index-offset-unaligned`, `render.error.memory-barrier-after-fragment` |

## Compute and blit encoders

| Behavior | Cases |
| --- | --- |
| `dispatchThreadgroups:` and position builtins | `compute.thread-positions` |
| `dispatchThreads:` with non-uniform threadgroups | `compute.dispatch-threads-nonuniform` |
| Indirect dispatch | `compute.indirect-dispatch` |
| Compute stage-in | `compute.stage-in-region` |
| Threadgroup memory | `compute.threadgroup-memory-length`, `compute.threadgroup-atomics` |
| `setBytes:` | `compute.set-bytes` |
| Concurrent dispatch with memory barriers | `compute.dispatch-type-concurrent` |
| Integer operations, conversions, single-precision builtins, atomics | `compute.integer-ops`, `compute.conversions`, `compute.float-builtins`, `compute.atomics` |
| Buffer copies and fills | `blit.copy-buffer`, `blit.copy-buffer-unaligned`, `blit.fill-buffer` |
| Buffer and texture copies with pitches, slices, faces, levels, 3D, 1D | `blit.buffer-to-texture-2d`, `blit.buffer-to-texture-3d`, `blit.slices-and-levels`, `blit.texture-to-buffer` |
| Texture copies: regions, whole textures, slice and level counts, multisample | `blit.texture-to-texture-region`, `blit.texture-to-texture-whole`, `blit.texture-to-texture-multisample` |
| Depth and stencil aspects, PVRTC row option, compressed blocks | `format.Depth32Float_Stencil8.upload-read`, `format.<format>.blocks` |
| Content optimization hints | `blit.optimize-contents` |
| Indirect command buffers: execution, ranges, reset, copy, optimize | `icb.compute`, `icb.render`, `icb.blit-copy-optimize-reset` |
| Misuse | `blit.copy-buffer-unaligned-macos`, `blit.texture-to-texture-format-mismatch` |

## Pipelines, libraries and archives

| Behavior | Cases |
| --- | --- |
| Libraries from source (sync, async), compile errors, compile options, math mode | `library.source-sync`, `library.source-async`, `library.source-compile-error`, `library.compile-options`, `library.math-mode` |
| Libraries from data and URLs, default library | `library.data`, `library.data-invalid`, `library.url-missing`, `library.default-absent` |
| Dynamic and stitched libraries | `library.dynamic`, `library.stitched` |
| Functions, constants, descriptors, reflection | `function.missing`, `function.constants`, `function.descriptor`, `function.reflection` |
| Pipeline creation (sync, async), reflection, errors | `pipeline.compute-reflection`, `pipeline.render-reflection`, `pipeline.compute-async`, `pipeline.render-async`, `pipeline.render-errors` |
| Pipeline descriptor fields and state properties | `pipeline.descriptor-defaults`, `pipeline.state-properties` |
| Binary archives | `archive.round-trip`, `archive.invalid-file` |
| Argument buffers | `argument-buffer.encoder-from-function`, `argument-buffer.encoder-from-descriptors`, `argument-buffer.arrays`, `argument-buffer.nested`, `argument-buffer.gpu-address` |

## Drawables

| Behavior | Cases |
| --- | --- |
| `CAMetalLayer` defaults, drawable textures, rendering and presenting, pool exhaustion | `drawable.layer-defaults`, `drawable.render-present`, `drawable.framebuffer-only`, `drawable.pool-exhaustion` |

## Validation

The `validation.` cases perform one misuse each, in both variants, and
record nil results, `NSError` domains and codes, Objective-C exceptions and
the process outcome: `validation.buffer-length-zero`,
`validation.buffer-length-over-max`, `validation.texture-width-zero`,
`validation.texture-width-over-max`, `validation.texture-mipmap-count-over-max`,
`validation.texture-multisample-with-mipmaps`,
`validation.texture-unsupported-sample-count`,
`validation.sampler-anisotropy-zero`,
`validation.texture-view-without-format-view-usage`,
`validation.compute-pipeline-from-vertex-function`,
`validation.render-pipeline-without-vertex-function`,
`validation.render-pipeline-blending-integer-format`,
`validation.function-constant-type-mismatch`,
`validation.threadgroup-above-pipeline-limit`,
`validation.set-buffer-index-over-limit`, `validation.set-bytes-over-4k`,
`validation.constant-buffer-offset-misaligned`,
`validation.dispatch-without-pipeline`, `validation.draw-without-pipeline`,
`validation.render-target-without-usage`,
`validation.texture-write-without-shader-write`,
`validation.encoder-use-after-end-encoding`,
`validation.encoder-on-committed-buffer`,
`validation.second-encoder-while-open`, `validation.private-buffer-contents`,
`validation.private-texture-get-bytes`,
`validation.private-texture-replace-region` and
`validation.private-texture-initial-contents`. The misuse cases of the other
areas are listed in their tables.

## System-client SPI

The driver SPI that QuartzCore, CoreImage, Metal Performance Shaders,
AppleMetalGLRenderer and WebKit send. The answers are the GPU driver's, so
they carry the device rule; Metal's handling of the work around them is
compared exactly.

| Behavior | Cases |
| --- | --- |
| `vendorName`, `familyName`, `productName` | `spi.device-names` |
| Command queue GPU priority and background GPU priority | `spi.queue-priority` |
| Command buffer cache hints over resource groups | `spi.cache-hints` |
| `imageblockSampleLength` and `staticThreadgroupMemoryLength` of a render pipeline without tile functions | `spi.render-pipeline-answers` |
| `commitAndWaitUntilSubmitted`, kernel start and end times across the command buffers of a queue | `spi.commit-and-wait-until-submitted` |
| A memoryless depth attachment on a device that reports `supportsMemorylessRenderTargets` | `spi.memoryless-render-target` |
| A raster order group on a fragment output, with the device's `rasterOrderGroupsSupported` answer | `spi.raster-order-groups` |
| A texture layout (`newTextureLayoutWithDescriptor:isHeapOrBufferBacked:`): its size and alignment, and its copies into and out of texture memory for every slice and level of 2D, 2D array, cube, 3D and ASTC textures and a sub-region (`copyFromLinearBytes:...`, `copyFromTextureMemory:...`) | `spi.texture-layout-copies` |
| `isComplete` and `waitUntilComplete` of tracked buffers used by encoders before and after an event wait | `spi.resource-complete-per-encoder` |

## Not covered

| Behavior | Reason |
| --- | --- |
| Private device SPI outside the system-client list, `registryID` | Only the SPI the system frameworks send is covered; `registryID` changes between boots |
| GPU role, `setOwnerWithIdentity:` | Private, with no observable effect on results |
| `MTLCaptureManager`, statistic counter sets, deadline profiles, performance-state assertions | No observable result; the measured device advertises only the timestamp counter set |
| Shared event handles across processes | Only the in-process handle round trip is covered |
| Command buffer error status with a defined cause | Every documented cause (timeout, page fault, out of memory) needs a GPU hang or fault |
| Private YCbCr pixel formats and the texture-memory SPI that system frameworks use | Private interfaces |
| Shader cache directories and cache identifiers | Private, not observable through the public API |
| Multiple viewports and scissors, `[[viewport_array_index]]` | No case yet |
| `alphaToOne`, stencil reference output from a fragment function, `[[point_coord]]` values | No case yet |
| `textureBarrier` | Deprecated and macOS-only |
| Depth bias on lines and points | No case yet |
| Sparse heaps and sparse textures | Only the tile-size queries are covered |
| Indirect command buffers encoded from shaders, argument encoders for indirect command buffers | No case yet; CPU encoding is covered |
| Tile shaders, imageblocks, mesh and amplification shaders, rasterization rate maps | No case yet; not part of the Apple3 feature set |
| Sampler LOD bias and minimum/maximum reduction | Listed for Apple10 and later; the reference device is Apple9 |
| Texture writes with `-ftexture-write-rounding-mode=rte` | The Metal Feature Set Tables do not list which families support it; native rounding is recorded with `none` |
| Compute kernels that call `sample` with `min_lod_clamp` | On the reference OS build the shader compiler service crashes while building the pipeline; the behavior is covered in a fragment function |
| Composition of a texture view swizzle with a descriptor swizzle | Metal does not define it |
| Accuracy of every builtin over its whole domain | A compact set is covered by `compute.float-builtins`; exhaustive measurement belongs to the numerics tests of the shader compiler |

Results that Metal leaves implementation-defined are covered with `none`
rules: line rasterization, edge ties, interpolation of non-flat varyings,
linear and anisotropic filtering, generated mipmaps, multisample color
resolve, alpha to coverage, depth bias values, decoded compressed texels,
extended-range and 4:2:2 decoding, render-target conversions to values that
need rounding, and native rounding of floating-point texture writes.
