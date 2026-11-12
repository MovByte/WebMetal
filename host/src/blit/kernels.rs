//! The GPU passes blits need where WebGPU has no command of its own, all exact:
//!
//! - `encode_rows`: moves rows of bytes between buffers (or fills them with one byte value) at any byte offset and
//!   pitch. WebGPU's `copyBufferToBuffer` moves whole 4-byte words and `clearBuffer` writes zeros only (WebGPU section
//!   13.2, `tables/limits.toml` question `6.6-clear-buffer`); Metal's iOS blits take any offset, size and byte value
//!   (`copy(from:sourceOffset:to:destinationOffset:size:)`, `fill(buffer:range:value:)`). Each invocation owns one
//!   destination word and writes back only the bytes inside the rows, so the bytes beside them keep their values.
//! - `encode_swap10`: exchanges the red and blue fields of 32-bit texels in place, the conversion between Metal's
//!   BGR10A2 bytes and the `rgb10a2unorm` texels that hold them ([`crate::format::packed::swap_red_blue_10`]).
//! - `encode_depth_write`: writes depth values into a depth aspect through a render pass, the only way WebGPU writes
//!   `depth32float` (its depth aspect is not a copy destination, WebGPU section 26.1.2).
//! - `encode_mip_level`: renders one mipmap level from the level above it with a linear filter.
//!
//! The pipelines are made on first use and kept in [`Kernels`], which the memory module holds for its device.

use std::cell::RefCell;

use crate::gpu::idl::{
    AutoLayoutMode, BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding, BufferDescriptor, BufferUsage,
    ColorTargetState, CompareFunction, ComputePassDescriptor, ComputePipelineDescriptor, DepthStencilState, Extent3D,
    Extent3DDict, FragmentState, LoadOp, Origin3D, Origin3DDict, PipelineLayoutOrAuto, ProgrammableStage,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
    SamplerDescriptor, ShaderModuleDescriptor, StoreOp, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureFormat, TextureOrView, TextureUsage, TextureViewDescriptor, TextureViewDimension,
    VertexState,
};
use crate::gpu::num::Float32;
use crate::gpu::{
    Api, BindingCommands as _, Buffer as _, CommandEncoder as _, ComputePassEncoder as _, Device as _,
    MappedRange as _, PipelineBase as _, Queue as _, RenderCommands as _, RenderPassEncoder as _, Size64, Texture as _,
};
use crate::mem::calls::Calls;
use crate::mem::{MemError, Memory};

/// Invocations per workgroup of the compute kernels.
const GROUP: u64 = 64;

/// Workgroups per dispatch dimension every WebGPU device allows (`maxComputeWorkgroupsPerDimension`, WebGPU section
/// 3.6.2 default).
const MAX_GROUPS: u64 = 65535;

const REPACK: &str = "
struct Params {
    src_byte: u32,
    dst_byte: u32,
    row_bytes: u32,
    rows: u32,
    images: u32,
    src_row_pitch: u32,
    src_image_pitch: u32,
    dst_row_pitch: u32,
    dst_image_pitch: u32,
    first_word: u32,
    words: u32,
    stride: u32,
    fill: u32,
    value: u32,
    pad0: u32,
    pad1: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn source_byte(i: u32) -> u32 {
    return (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let k = id.y * p.stride + id.x;
    if (k >= p.words) {
        return;
    }
    let w = p.first_word + k;
    var word = dst[w];
    for (var b = 0u; b < 4u; b++) {
        let g = w * 4u + b;
        if (g < p.dst_byte) {
            continue;
        }
        let rel = g - p.dst_byte;
        var image = 0u;
        var within = rel;
        if (p.images > 1u) {
            image = rel / p.dst_image_pitch;
            within = rel % p.dst_image_pitch;
        }
        if (image >= p.images) {
            continue;
        }
        var row = 0u;
        var col = within;
        if (p.rows > 1u) {
            row = within / p.dst_row_pitch;
            col = within % p.dst_row_pitch;
        }
        if (row >= p.rows || col >= p.row_bytes) {
            continue;
        }
        var v = p.value & 0xffu;
        if (p.fill == 0u) {
            v = source_byte(p.src_byte + image * p.src_image_pitch + row * p.src_row_pitch + col);
        }
        let shift = b * 8u;
        word = (word & ~(0xffu << shift)) | (v << shift);
    }
    dst[w] = word;
}
";

const SWAP10: &str = "
struct Params {
    first_word: u32,
    words: u32,
    stride: u32,
    pad: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let k = id.y * p.stride + id.x;
    if (k >= p.words) {
        return;
    }
    let w = data[p.first_word + k];
    data[p.first_word + k] = (w & 0xc00ffc00u) | ((w & 0x3ffu) << 20u) | ((w >> 20u) & 0x3ffu);
}
";

/// A triangle that covers the whole viewport, for the render passes.
const FULLSCREEN: &str = "
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = f32(i >> 1u) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}
";

/// The fragment of the depth write: the bits of the value for this texel, as a float.
const DEPTH: &str = "
struct Params {
    origin: vec2u,
    pad: vec2u,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var values: texture_2d<u32>;

@fragment
fn fs(@builtin(position) position: vec4f) -> @builtin(frag_depth) f32 {
    let at = vec2u(position.xy) - p.origin;
    return bitcast<f32>(textureLoad(values, at, 0).r);
}
";

/// The fragment of a mipmap level of a 2D slice: the source level sampled with a linear filter at the centre of the
/// destination texel.
const MIP_2D: &str = "
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear: sampler;

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let size = vec2f(textureDimensions(source)) * 0.5;
    let extent = max(floor(size), vec2f(1.0));
    return textureSampleLevel(source, linear, position.xy / extent, 0.0);
}
";

/// The fragment of a mipmap level of one depth slice of a 3D texture.
const MIP_3D: &str = "
struct Params {
    slice: f32,
    depth: f32,
    pad0: f32,
    pad1: f32,
}
@group(0) @binding(0) var source: texture_3d<f32>;
@group(0) @binding(1) var linear: sampler;
@group(0) @binding(2) var<uniform> p: Params;

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let size = vec2f(textureDimensions(source).xy) * 0.5;
    let extent = max(floor(size), vec2f(1.0));
    let w = (p.slice + 0.5) / p.depth;
    return textureSampleLevel(source, linear, vec3f(position.xy / extent, w), 0.0);
}
";

type Compute<A> = (<A as Api>::ComputePipeline, <A as Api>::BindGroupLayout);
type Render<A> = (<A as Api>::RenderPipeline, <A as Api>::BindGroupLayout);
/// A mipmap pipeline by (view format, 3D) with its sampler.
type Mip<A> = ((TextureFormat, bool), Render<A>, <A as Api>::Sampler);

/// The blit passes' pipelines on one device.
pub struct Kernels<A: Api> {
    repack: RefCell<Option<Compute<A>>>,
    swap10: RefCell<Option<Compute<A>>>,
    depth: RefCell<Vec<(TextureFormat, Render<A>)>>,
    mips: RefCell<Vec<Mip<A>>>,
    /// A 4-byte buffer bound as the source of a fill, which reads no source.
    dummy: RefCell<Option<A::Buffer>>,
}

impl<A: Api> Default for Kernels<A> {
    fn default() -> Kernels<A> {
        Kernels {
            repack: RefCell::new(None),
            swap10: RefCell::new(None),
            depth: RefCell::new(Vec::new()),
            mips: RefCell::new(Vec::new()),
            dummy: RefCell::new(None),
        }
    }
}

fn gpu_err(op: &'static str) -> impl Fn(crate::gpu::Exception) -> MemError {
    move |e| MemError::gpu(op, e)
}

fn size64(v: u64, op: &'static str) -> Result<Size64, MemError> {
    Size64::new(v).map_err(gpu_err(op))
}

fn origin(x: u32, y: u32, z: u32) -> Origin3D {
    Origin3D::Origin3DDict(Origin3DDict { x, y, z })
}

fn extent(w: u32, h: u32, d: u32) -> Extent3D {
    Extent3D::Extent3DDict(Extent3DDict { width: w, height: h, depth_or_array_layers: d })
}

fn compute_pipeline<A: Api>(g: &impl Calls<A>, label: &str, code: &str) -> Compute<A> {
    let module = g.call("GPUDevice.createShaderModule", label, |d| {
        d.create_shader_module(&ShaderModuleDescriptor::new(code.into()))
    });
    let mut desc = ComputePipelineDescriptor::new(
        PipelineLayoutOrAuto::AutoLayoutMode(AutoLayoutMode::Auto),
        ProgrammableStage::new(&module),
    );
    desc.label = label.into();
    let pipeline = g.call("GPUDevice.createComputePipeline", label, |d| d.create_compute_pipeline(&desc));
    let layout = g.call("GPUPipelineBase.getBindGroupLayout", label, |_| pipeline.get_bind_group_layout(0));
    (pipeline, layout)
}

/// A uniform buffer holding `words`, made mapped so that it needs no queue write.
fn uniforms<A: Api>(g: &impl Calls<A>, label: &str, words: &[u32]) -> Result<A::Buffer, MemError> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer_with(g, label, &bytes, BufferUsage::UNIFORM)
}

/// A buffer of `usage` holding `bytes` (padded with zeros to whole words), made mapped at creation.
pub(crate) fn buffer_with<A: Api>(
    g: &impl Calls<A>,
    label: &str,
    bytes: &[u8],
    usage: u32,
) -> Result<A::Buffer, MemError> {
    let size = (bytes.len() as u64).next_multiple_of(4).max(4);
    let mut d = BufferDescriptor::new(size64(size, "createBuffer")?, usage);
    d.mapped_at_creation = true;
    d.label = label.into();
    let b = g.call("GPUDevice.createBuffer", label, |dev| dev.create_buffer(&d)).map_err(gpu_err("createBuffer"))?;
    let m = b.get_mapped_range(Size64::new_const(0), None).map_err(gpu_err("GPUBuffer.getMappedRange"))?;
    m.write(0, bytes).map_err(gpu_err("ArrayBuffer write"))?;
    b.unmap();
    Ok(b)
}

/// A buffer of `size` bytes (rounded up to whole words) for the passes' intermediate bytes.
pub(crate) fn scratch<A: Api>(g: &impl Calls<A>, label: &str, size: u64, usage: u32) -> Result<A::Buffer, MemError> {
    let mut d = BufferDescriptor::new(size64(size.next_multiple_of(4).max(4), "createBuffer")?, usage);
    d.label = label.into();
    g.call("GPUDevice.createBuffer", label, |dev| dev.create_buffer(&d)).map_err(gpu_err("createBuffer"))
}

/// The dispatch size covering `words` invocations: x workgroups, y workgroups, and invocations per row of the grid.
fn grid(words: u64) -> Result<(u32, u32, u32), MemError> {
    let groups = words.div_ceil(GROUP).max(1);
    let x = groups.min(MAX_GROUPS);
    let y = groups.div_ceil(x);
    if y > MAX_GROUPS {
        return Err(MemError::host(format!("a pass over {words} words exceeds the dispatch size every device allows")));
    }
    Ok((x as u32, y as u32, (x * GROUP) as u32))
}

/// The byte layout of rows the repack moves: `images` images of `rows` rows of `row_bytes` bytes, at the pitches of the
/// source and of the destination. A pitch that is not used (one row, one image) may be anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rows {
    pub row_bytes: u64,
    pub rows: u64,
    pub images: u64,
    pub src_row_pitch: u64,
    pub src_image_pitch: u64,
    pub dst_row_pitch: u64,
    pub dst_image_pitch: u64,
}

impl Rows {
    /// One run of `len` bytes.
    pub(crate) fn run(len: u64) -> Rows {
        Rows {
            row_bytes: len,
            rows: 1,
            images: 1,
            src_row_pitch: 0,
            src_image_pitch: 0,
            dst_row_pitch: 0,
            dst_image_pitch: 0,
        }
    }

    fn src_span(&self) -> u64 {
        (self.images - 1) * self.src_image_pitch + (self.rows - 1) * self.src_row_pitch + self.row_bytes
    }

    fn dst_span(&self) -> u64 {
        (self.images - 1) * self.dst_image_pitch + (self.rows - 1) * self.dst_row_pitch + self.row_bytes
    }

    fn is_empty(&self) -> bool {
        self.row_bytes == 0 || self.rows == 0 || self.images == 0
    }
}

/// Where the repack's bytes come from.
pub(crate) enum Source<'a, A: Api> {
    /// The bytes of a buffer from a byte offset. It must not be the destination buffer: WebGPU does not bind one
    /// buffer as both a read-only and a writable storage buffer of one dispatch (WebGPU section 3.4.5).
    Buffer(&'a A::Buffer, u64),
    /// One byte value.
    Fill(u8),
}

impl<A: Api> Clone for Source<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: Api> Copy for Source<'_, A> {}

/// The binding window of a range: an offset aligned for storage bindings and a size of whole words.
fn window(offset: u64, span: u64, buffer_size: u64, align: u64) -> (u64, u64) {
    let base = offset / align * align;
    let end = (offset + span).next_multiple_of(4).min(buffer_size);
    (base, end - base)
}

/// The largest storage binding a repack window uses: the device's limit, kept below 2^30 so that every byte index the
/// kernel forms fits in a u32.
fn max_window<A: Api>(mem: &Memory<A>) -> u64 {
    mem.limits.max_storage_buffer_binding_size.unwrap_or(128 << 20).min(1 << 30) & !3
}

fn storage_align<A: Api>(mem: &Memory<A>) -> u64 {
    u64::from(mem.limits.min_storage_buffer_offset_alignment.unwrap_or(256))
}

/// Encode, as compute passes on `encoder`, the move of `rows` from `src` into `dst` at byte offset `dst_offset`. The
/// destination's bytes outside the rows keep their values. `encoder` must have no pass open.
pub(crate) fn encode_rows<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    src: Source<'_, A>,
    (dst, dst_offset): (&A::Buffer, u64),
    rows: Rows,
) -> Result<(), MemError> {
    if rows.is_empty() {
        return Ok(());
    }
    let limit = max_window(mem);
    let align = storage_align(mem);
    // A window may start up to `align` - 4 bytes before the range and end up to 3 bytes after it.
    let fits = |r: &Rows, src_off: u64, dst_off: u64| {
        let s = match src {
            Source::Buffer(..) => r.src_span() + src_off % align + 4,
            Source::Fill(_) => 0,
        };
        s <= limit && r.dst_span() + dst_off % align + 4 <= limit
    };
    let src_offset = match src {
        Source::Buffer(_, o) => o,
        Source::Fill(_) => 0,
    };
    // Split the rows until each part's windows fit the binding limit: by images, then rows, then bytes of one row.
    let mut work = vec![(rows, src_offset, dst_offset)];
    while let Some((r, so, d)) = work.pop() {
        if fits(&r, so, d) {
            dispatch_rows(mem, g, encoder, src, so, (dst, d), r)?;
            continue;
        }
        let (a, b, sa, da) = if r.images > 1 {
            let k = r.images / 2;
            (Rows { images: k, ..r }, Rows { images: r.images - k, ..r }, k * r.src_image_pitch, k * r.dst_image_pitch)
        } else if r.rows > 1 {
            let k = r.rows / 2;
            (Rows { rows: k, ..r }, Rows { rows: r.rows - k, ..r }, k * r.src_row_pitch, k * r.dst_row_pitch)
        } else {
            let k = (r.row_bytes / 2).max(1);
            (Rows { row_bytes: k, ..r }, Rows { row_bytes: r.row_bytes - k, ..r }, k, k)
        };
        // The second part runs after the first: a word the two share sees the first part's bytes.
        work.push((b, so + sa, d + da));
        work.push((a, so, d));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_rows<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    src: Source<'_, A>,
    src_offset: u64,
    (dst, dst_offset): (&A::Buffer, u64),
    r: Rows,
) -> Result<(), MemError> {
    const LABEL: &str = "webmetal blit repack";
    let align = storage_align(mem);
    let k = &mem.kernels;
    if k.repack.borrow().is_none() {
        *k.repack.borrow_mut() = Some(compute_pipeline(g, LABEL, REPACK));
    }
    if k.dummy.borrow().is_none() {
        *k.dummy.borrow_mut() = Some(buffer_with(g, LABEL, &[0; 4], BufferUsage::STORAGE)?);
    }
    let dummy = k.dummy.borrow().clone().expect("made above");
    let (dst_base, dst_size) = window(dst_offset, r.dst_span(), dst.size(), align);
    let (src_buffer, src_base, src_size, fill, value) = match src {
        Source::Buffer(b, _) => {
            let (base, size) = window(src_offset, r.src_span(), b.size(), align);
            (b.clone(), base, size, 0, 0)
        }
        Source::Fill(v) => (dummy, 0, 4, 1, u32::from(v)),
    };
    let dst_byte = dst_offset - dst_base;
    let first_word = dst_byte / 4;
    let words = (dst_byte + r.dst_span()).div_ceil(4) - first_word;
    let (gx, gy, stride) = grid(words)?;
    let pitch = |p: u64, n: u64| if n > 1 { p as u32 } else { 0 };
    let params = uniforms(
        g,
        LABEL,
        &[
            (src_offset - src_base) as u32,
            dst_byte as u32,
            r.row_bytes as u32,
            r.rows as u32,
            r.images as u32,
            pitch(r.src_row_pitch, r.rows),
            pitch(r.src_image_pitch, r.images),
            pitch(r.dst_row_pitch, r.rows),
            pitch(r.dst_image_pitch, r.images),
            first_word as u32,
            words as u32,
            stride,
            fill,
            value,
            0,
            0,
        ],
    )?;
    let repack = k.repack.borrow();
    let (pipeline, layout) = repack.as_ref().expect("made above");
    let entries = vec![
        BindGroupEntry::new(0, BindingResource::Buffer(&params)),
        BindGroupEntry::new(
            1,
            BindingResource::BufferBinding(BufferBinding {
                buffer: &src_buffer,
                offset: size64(src_base, "createBindGroup")?,
                size: Some(size64(src_size, "createBindGroup")?),
            }),
        ),
        BindGroupEntry::new(
            2,
            BindingResource::BufferBinding(BufferBinding {
                buffer: dst,
                offset: size64(dst_base, "createBindGroup")?,
                size: Some(size64(dst_size, "createBindGroup")?),
            }),
        ),
    ];
    let group =
        g.call("GPUDevice.createBindGroup", LABEL, |d| d.create_bind_group(&BindGroupDescriptor::new(layout, entries)));
    run_compute(g, encoder, LABEL, pipeline, &group, (gx, gy))
}

fn run_compute<A: Api>(
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    label: &str,
    pipeline: &A::ComputePipeline,
    group: &A::BindGroup,
    (gx, gy): (u32, u32),
) -> Result<(), MemError> {
    let mut pd = ComputePassDescriptor::new();
    pd.label = label.into();
    let pass = g
        .call("GPUCommandEncoder.beginComputePass", label, |_| encoder.begin_compute_pass(&pd))
        .map_err(gpu_err("GPUCommandEncoder.beginComputePass"))?;
    g.call("GPUComputePassEncoder.setPipeline", label, |_| pass.set_pipeline(pipeline));
    g.call("GPUComputePassEncoder.setBindGroup", label, |_| pass.set_bind_group(0, Some(group), &[]));
    g.call("GPUComputePassEncoder.dispatchWorkgroups", label, |_| pass.dispatch_workgroups(gx, gy, 1));
    g.call("GPUComputePassEncoder.end", label, |_| pass.end());
    Ok(())
}

/// Encode the exchange of the red and blue fields of the `words` 32-bit texels of `buffer` from byte `offset` (a
/// multiple of 4).
pub(crate) fn encode_swap10<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    buffer: &A::Buffer,
    offset: u64,
    words: u64,
) -> Result<(), MemError> {
    const LABEL: &str = "webmetal BGR10A2 channel swap";
    debug_assert!(offset.is_multiple_of(4));
    if words == 0 {
        return Ok(());
    }
    let k = &mem.kernels;
    if k.swap10.borrow().is_none() {
        *k.swap10.borrow_mut() = Some(compute_pipeline(g, LABEL, SWAP10));
    }
    let limit = max_window(mem);
    let align = storage_align(mem);
    let mut at = offset;
    let end = offset + words * 4;
    while at < end {
        let base = at / align * align;
        let n = ((end - at) / 4).min((limit - (at - base)) / 4);
        let size = (at - base) + n * 4;
        let (gx, gy, stride) = grid(n)?;
        let params = uniforms(g, LABEL, &[((at - base) / 4) as u32, n as u32, stride, 0])?;
        let swap = k.swap10.borrow();
        let (pipeline, layout) = swap.as_ref().expect("made above");
        let entries = vec![
            BindGroupEntry::new(0, BindingResource::Buffer(&params)),
            BindGroupEntry::new(
                1,
                BindingResource::BufferBinding(BufferBinding {
                    buffer,
                    offset: size64(base, "createBindGroup")?,
                    size: Some(size64(size, "createBindGroup")?),
                }),
            ),
        ];
        let group = g.call("GPUDevice.createBindGroup", LABEL, |d| {
            d.create_bind_group(&BindGroupDescriptor::new(layout, entries))
        });
        run_compute(g, encoder, LABEL, pipeline, &group, (gx, gy))?;
        at += n * 4;
    }
    Ok(())
}

fn render_pipeline<A: Api>(
    g: &impl Calls<A>,
    label: &str,
    fragment: &str,
    depth: Option<TextureFormat>,
    color: Option<TextureFormat>,
) -> Result<Render<A>, MemError> {
    let code = format!("{FULLSCREEN}{fragment}");
    let module =
        g.call("GPUDevice.createShaderModule", label, |d| d.create_shader_module(&ShaderModuleDescriptor::new(code)));
    let mut vertex = VertexState::new(&module);
    vertex.entry_point = Some("vs".into());
    let mut desc = RenderPipelineDescriptor::new(PipelineLayoutOrAuto::AutoLayoutMode(AutoLayoutMode::Auto), vertex);
    desc.label = label.into();
    let mut frag = FragmentState::new(&module, color.map(|f| vec![Some(ColorTargetState::new(f))]).unwrap_or_default());
    frag.entry_point = Some("fs".into());
    desc.fragment = Some(frag);
    if let Some(f) = depth {
        let mut ds = DepthStencilState::new(f);
        ds.depth_write_enabled = Some(true);
        ds.depth_compare = Some(CompareFunction::Always);
        // The stencil aspect of a combined format keeps its values.
        ds.stencil_write_mask = 0;
        desc.depth_stencil = Some(ds);
    }
    let pipeline = g
        .call("GPUDevice.createRenderPipeline", label, |d| d.create_render_pipeline(&desc))
        .map_err(gpu_err("GPUDevice.createRenderPipeline"))?;
    let layout = g.call("GPUPipelineBase.getBindGroupLayout", label, |_| pipeline.get_bind_group_layout(0));
    Ok((pipeline, layout))
}

/// A depth write: the region of one subresource of a depth or depth-stencil texture and its values.
pub(crate) struct DepthWrite<'a, A: Api> {
    pub texture: &'a A::Texture,
    pub level: u32,
    /// The array layer (slice or cube face).
    pub layer: u32,
    pub origin: (u32, u32),
    pub size: (u32, u32),
    /// The IEEE binary32 bits of each value, row by row.
    pub bits: &'a [u32],
}

/// Is `bits` a depth value the render-pass write stores unchanged? `+0.0` and the normal values of `[0, 1]`: the depth
/// output is clamped to the viewport's depth range `[0, 1]` (WebGPU section 23.2.6, "fragment processing"), which may
/// change the sign of a zero, and nothing promises that a subnormal or a NaN reaches the attachment unchanged.
pub(crate) fn depth_value_stored_exactly(bits: u32) -> bool {
    let v = f32::from_bits(bits);
    bits == 0 || (v.is_normal() && v > 0.0 && v <= 1.0)
}

/// The binary32 bits of the depth values in `bytes`, each checked with `depth_value_stored_exactly`: a value the
/// render-pass write would not store unchanged is refused.
pub(crate) fn depth_bits(call: &'static str, bytes: &[u8]) -> Result<Vec<u32>, MemError> {
    let (words, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(MemError::invalid(format!("{call}: depth bytes are not whole 32-bit values")));
    }
    words
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .map(|b| {
            if depth_value_stored_exactly(b) {
                Ok(b)
            } else {
                Err(MemError::unimplemented(format!(
                    "{call}: depth value {b:#010x} is not +0 or a normal value of [0, 1]; WebGPU writes depth32float \
                     only through a depth output clamped to [0, 1], which does not keep it"
                )))
            }
        })
        .collect()
}

/// Encode a render pass that writes `w.bits` into the depth aspect of a region of one subresource; a stencil aspect
/// keeps its values. The texture must allow `RENDER_ATTACHMENT`, and every value must satisfy
/// `depth_value_stored_exactly` (the caller checks, and refuses otherwise).
pub(crate) fn encode_depth_write<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    w: &DepthWrite<'_, A>,
) -> Result<(), MemError> {
    const LABEL: &str = "webmetal depth write";
    debug_assert!(w.bits.iter().all(|&b| depth_value_stored_exactly(b)));
    let format = w.texture.format();
    let stencil = match format {
        TextureFormat::Depth32float => false,
        TextureFormat::Depth32floatStencil8 => true,
        f => return Err(MemError::unimplemented(format!("a depth write into {}", f.as_str()))),
    };
    if w.texture.usage() & TextureUsage::RENDER_ATTACHMENT == 0 {
        return Err(MemError::host(format!(
            "writing a {} texture needs a render pass, and this host cannot render to it",
            format.as_str()
        )));
    }
    let k = &mem.kernels;
    if !k.depth.borrow().iter().any(|(f, _)| *f == format) {
        let p = render_pipeline(g, LABEL, DEPTH, Some(format), None)?;
        k.depth.borrow_mut().push((format, p));
    }
    let (width, height) = w.size;
    // The values, as a uint texture the fragment reads texel by texel.
    let mut td = TextureDescriptor::new(
        extent(width, height, 1),
        TextureFormat::R32uint,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    );
    td.label = LABEL.into();
    let values = g
        .call("GPUDevice.createTexture", LABEL, |d| d.create_texture(&td))
        .map_err(gpu_err("GPUDevice.createTexture"))?;
    let bytes: Vec<u8> = w.bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let layout = TexelCopyBufferLayout {
        offset: Size64::new_const(0),
        bytes_per_row: Some(width * 4),
        rows_per_image: Some(height),
    };
    // The texture is new and only this pass reads it, so the queue write needs no ordering with the encoder.
    g.call("GPUQueue.writeTexture", LABEL, |_| {
        g.queue().write_texture(&TexelCopyTextureInfo::new(&values), &bytes, &layout, &extent(width, height, 1))
    })
    .map_err(gpu_err("GPUQueue.writeTexture"))?;
    let values_view = g
        .call("GPUTexture.createView", LABEL, |_| values.create_view(&TextureViewDescriptor::new()))
        .map_err(gpu_err("GPUTexture.createView"))?;
    let params = uniforms(g, LABEL, &[w.origin.0, w.origin.1, 0, 0])?;
    let depth = k.depth.borrow();
    let (_, (pipeline, bgl)) = depth.iter().find(|(f, _)| *f == format).expect("made above");
    let entries = vec![
        BindGroupEntry::new(0, BindingResource::Buffer(&params)),
        BindGroupEntry::new(1, BindingResource::TextureView(&values_view)),
    ];
    let group =
        g.call("GPUDevice.createBindGroup", LABEL, |d| d.create_bind_group(&BindGroupDescriptor::new(bgl, entries)));
    let mut vd = TextureViewDescriptor::new();
    vd.dimension = Some(TextureViewDimension::D2);
    vd.base_mip_level = w.level;
    vd.mip_level_count = Some(1);
    vd.base_array_layer = w.layer;
    vd.array_layer_count = Some(1);
    vd.aspect = TextureAspect::All;
    let target = g
        .call("GPUTexture.createView", LABEL, |_| w.texture.create_view(&vd))
        .map_err(gpu_err("GPUTexture.createView"))?;
    let mut ds = RenderPassDepthStencilAttachment::new(TextureOrView::TextureView(&target));
    ds.depth_load_op = Some(LoadOp::Load);
    ds.depth_store_op = Some(StoreOp::Store);
    if stencil {
        ds.stencil_load_op = Some(LoadOp::Load);
        ds.stencil_store_op = Some(StoreOp::Store);
    }
    let mut rp = RenderPassDescriptor::new(Vec::new());
    rp.label = LABEL.into();
    rp.depth_stencil_attachment = Some(ds);
    let pass = g
        .call("GPUCommandEncoder.beginRenderPass", LABEL, |_| encoder.begin_render_pass(&rp))
        .map_err(gpu_err("GPUCommandEncoder.beginRenderPass"))?;
    let f = |v: u32| Float32::new(v as f32).expect("finite");
    let (x, y) = w.origin;
    g.call("GPURenderPassEncoder.setViewport", LABEL, |_| {
        pass.set_viewport(f(x), f(y), f(width), f(height), Float32::new_const(0.0), Float32::new_const(1.0))
    });
    g.call("GPURenderPassEncoder.setScissorRect", LABEL, |_| pass.set_scissor_rect(x, y, width, height));
    g.call("GPURenderPassEncoder.setPipeline", LABEL, |_| pass.set_pipeline(pipeline));
    g.call("GPURenderPassEncoder.setBindGroup", LABEL, |_| pass.set_bind_group(0, Some(&group), &[]));
    g.call("GPURenderPassEncoder.draw", LABEL, |_| pass.draw(3, 1, 0, 0));
    g.call("GPURenderPassEncoder.end", LABEL, |_| pass.end());
    Ok(())
}

/// One mipmap level of one slice to generate.
pub(crate) struct MipLevel<'a, A: Api> {
    pub texture: &'a A::Texture,
    /// The format the levels are viewed in (the sRGB format of an sRGB texture, so that filtering is in linear space).
    pub format: TextureFormat,
    /// The level to write; it is made from `level - 1`.
    pub level: u32,
    /// The array layer, or for a 3D texture the depth slice of `level`.
    pub layer: u32,
    pub three_d: bool,
    /// The extent of `level`.
    pub size: (u32, u32, u32),
}

/// Encode the render pass that makes one slice of a mipmap level from the level above it.
pub(crate) fn encode_mip_level<A: Api>(
    mem: &Memory<A>,
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    m: &MipLevel<'_, A>,
) -> Result<(), MemError> {
    const LABEL: &str = "webmetal mipmap generation";
    let k = &mem.kernels;
    let key = (m.format, m.three_d);
    if !k.mips.borrow().iter().any(|(kk, _, _)| *kk == key) {
        let p = render_pipeline(g, LABEL, if m.three_d { MIP_3D } else { MIP_2D }, None, Some(m.format))?;
        let mut sd = SamplerDescriptor::new();
        sd.label = LABEL.into();
        sd.mag_filter = crate::gpu::idl::FilterMode::Linear;
        sd.min_filter = crate::gpu::idl::FilterMode::Linear;
        let sampler = g.call("GPUDevice.createSampler", LABEL, |d| d.create_sampler(&sd));
        k.mips.borrow_mut().push((key, p, sampler));
    }
    let mips = k.mips.borrow();
    let (_, (pipeline, bgl), sampler) = mips.iter().find(|(kk, _, _)| *kk == key).expect("made above");
    // Each view takes the one usage it is bound with: a view in the texture's sRGB sibling format may lack a usage of
    // the texture's, such as storage (WebGPU section 6.2.1).
    let mut sv = TextureViewDescriptor::new();
    sv.format = Some(m.format);
    sv.usage = crate::gpu::idl::TextureUsage::TEXTURE_BINDING;
    sv.dimension = Some(if m.three_d { TextureViewDimension::D3 } else { TextureViewDimension::D2 });
    sv.base_mip_level = m.level - 1;
    sv.mip_level_count = Some(1);
    if !m.three_d {
        sv.base_array_layer = m.layer;
        sv.array_layer_count = Some(1);
    }
    let source = g
        .call("GPUTexture.createView", LABEL, |_| m.texture.create_view(&sv))
        .map_err(gpu_err("GPUTexture.createView"))?;
    let mut entries = vec![
        BindGroupEntry::new(0, BindingResource::TextureView(&source)),
        BindGroupEntry::new(1, BindingResource::Sampler(sampler)),
    ];
    let params;
    if m.three_d {
        params = uniforms(g, LABEL, &[(m.layer as f32).to_bits(), (m.size.2 as f32).to_bits(), 0, 0])?;
        entries.push(BindGroupEntry::new(2, BindingResource::Buffer(&params)));
    }
    let group =
        g.call("GPUDevice.createBindGroup", LABEL, |d| d.create_bind_group(&BindGroupDescriptor::new(bgl, entries)));
    let mut tv = TextureViewDescriptor::new();
    tv.format = Some(m.format);
    tv.usage = crate::gpu::idl::TextureUsage::RENDER_ATTACHMENT;
    tv.base_mip_level = m.level;
    tv.mip_level_count = Some(1);
    if m.three_d {
        tv.dimension = Some(TextureViewDimension::D3);
    } else {
        tv.dimension = Some(TextureViewDimension::D2);
        tv.base_array_layer = m.layer;
        tv.array_layer_count = Some(1);
    }
    let target = g
        .call("GPUTexture.createView", LABEL, |_| m.texture.create_view(&tv))
        .map_err(gpu_err("GPUTexture.createView"))?;
    let mut ca = RenderPassColorAttachment::new(TextureOrView::TextureView(&target), LoadOp::Load, StoreOp::Store);
    if m.three_d {
        ca.depth_slice = Some(m.layer);
    }
    let mut rp = RenderPassDescriptor::new(vec![Some(ca)]);
    rp.label = LABEL.into();
    let pass = g
        .call("GPUCommandEncoder.beginRenderPass", LABEL, |_| encoder.begin_render_pass(&rp))
        .map_err(gpu_err("GPUCommandEncoder.beginRenderPass"))?;
    g.call("GPURenderPassEncoder.setPipeline", LABEL, |_| pass.set_pipeline(pipeline));
    g.call("GPURenderPassEncoder.setBindGroup", LABEL, |_| pass.set_bind_group(0, Some(&group), &[]));
    g.call("GPURenderPassEncoder.draw", LABEL, |_| pass.draw(3, 1, 0, 0));
    g.call("GPURenderPassEncoder.end", LABEL, |_| pass.end());
    Ok(())
}

/// Encode the copy of a region between textures through a buffer of the destination encoder: for copies WebGPU does not
/// make directly between the two (one texture to itself).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_texture_copy_staged<A: Api>(
    g: &impl Calls<A>,
    encoder: &A::CommandEncoder,
    (src, src_level, src_origin): (&A::Texture, u32, (u32, u32, u32)),
    (dst, dst_level, dst_origin): (&A::Texture, u32, (u32, u32, u32)),
    size: (u32, u32, u32),
    aspect: TextureAspect,
    (bytes_per_block, block): (u32, (u32, u32)),
) -> Result<(), MemError> {
    const LABEL: &str = "webmetal staged texture copy";
    let blocks_wide = size.0.div_ceil(block.0);
    let blocks_high = size.1.div_ceil(block.1);
    let row = (u64::from(blocks_wide) * u64::from(bytes_per_block)).next_multiple_of(256);
    let total = row * u64::from(blocks_high) * u64::from(size.2);
    let d = BufferDescriptor::new(size64(total.max(4), "createBuffer")?, BufferUsage::COPY_SRC | BufferUsage::COPY_DST);
    let staging = g
        .call("GPUDevice.createBuffer", LABEL, |dev| dev.create_buffer(&d))
        .map_err(gpu_err("GPUDevice.createBuffer"))?;
    let mut s = TexelCopyTextureInfo::<A>::new(src);
    s.mip_level = src_level;
    s.origin = origin(src_origin.0, src_origin.1, src_origin.2);
    s.aspect = aspect;
    let mut b = crate::gpu::idl::TexelCopyBufferInfo::<A>::new(&staging);
    b.bytes_per_row = Some(row as u32);
    b.rows_per_image = Some(blocks_high);
    let e = extent(size.0, size.1, size.2);
    g.call("GPUCommandEncoder.copyTextureToBuffer", LABEL, |_| encoder.copy_texture_to_buffer(&s, &b, &e))
        .map_err(gpu_err("GPUCommandEncoder.copyTextureToBuffer"))?;
    let mut t = TexelCopyTextureInfo::<A>::new(dst);
    t.mip_level = dst_level;
    t.origin = origin(dst_origin.0, dst_origin.1, dst_origin.2);
    t.aspect = aspect;
    g.call("GPUCommandEncoder.copyBufferToTexture", LABEL, |_| encoder.copy_buffer_to_texture(&b, &t, &e))
        .map_err(gpu_err("GPUCommandEncoder.copyBufferToTexture"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_values_the_write_keeps() {
        for v in [0.0f32, 1.0, 0.5, f32::MIN_POSITIVE, 1.0 / 16_777_216.0] {
            assert!(depth_value_stored_exactly(v.to_bits()), "{v}");
        }
        for v in [-0.0f32, -0.5, 1.5, f32::NAN, f32::INFINITY, f32::MIN_POSITIVE / 2.0] {
            assert!(!depth_value_stored_exactly(v.to_bits()), "{v}");
        }
        assert!(depth_bits("t", &1.5f32.to_le_bytes()).is_err());
        assert_eq!(depth_bits("t", &0.25f32.to_le_bytes()).unwrap(), vec![0.25f32.to_bits()]);
    }

    #[test]
    fn windows_and_grids() {
        // A range at byte 300 of 10 bytes: bound from 256, to the word after its end.
        assert_eq!(window(300, 10, 4096, 256), (256, 56));
        // Clipped to the buffer.
        assert_eq!(window(4090, 6, 4096, 256), (3840, 256));
        assert_eq!(grid(1).unwrap(), (1, 1, 64));
        assert_eq!(grid(64 * 65535 + 1).unwrap(), (65535, 2, 65535 * 64));
    }

    #[test]
    fn row_spans() {
        let r = Rows {
            row_bytes: 11,
            rows: 5,
            images: 2,
            src_row_pitch: 256,
            src_image_pitch: 2048,
            dst_row_pitch: 13,
            dst_image_pitch: 100,
        };
        assert_eq!(r.src_span(), 2048 + 4 * 256 + 11);
        assert_eq!(r.dst_span(), 100 + 4 * 13 + 11);
        assert_eq!(Rows::run(7).dst_span(), 7);
    }
}
