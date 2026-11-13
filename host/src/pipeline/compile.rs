//! Calls into the compiler (AirWgsl `compile`) and the cache of what it produced.
//!
//! A compilation is determined by its input: the stage modules (by digest, which the registry has checked, with their
//! entry function, type and function constants), the [`PipelineDesc`] (everything else that changes the generated
//! code: the pipeline's kind and state and the variant's dynamic state) and the host's capabilities. [`key`] encodes
//! that input canonically, field by field with every struct destructured, so a field the compiler gains changes this
//! file before it can be left out of the key. Equal keys compile to equal results, so the cache returns the result of
//! the first compilation for every later one: the WGSL a hit returns is byte-equal to a fresh compilation's. Refusals
//! are cached like successes; the compiler has no state between calls, so a refused input stays refused.
//!
//! The cache is the executor's for its one device and lives as long as the executor; entries are dropped oldest
//! first once the WGSL they hold passes the budget ([`Compiler::new`]).

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use airwgsl::{
    AirModule, ArgumentBufferSlot, BufferSlot, ColorAttachment, Compiled, ComputeState, EntryPoint, FunctionConstant,
    HostCaps, PipelineDesc, PipelineKind, Refusal, RenderState, SamplerSlot, StageBindings, StageInputDescriptor,
    Tessellation, TextureSlot, TileState, VertexAttribute, VertexBufferLayout, VertexDescriptor,
};

use crate::gpu::idl::SUPPORTED_LIMITS;
use crate::gpu::{Api, Device};

/// The outcome of one compilation.
pub type Outcome = Rc<Result<Compiled, Refusal>>;

/// The capabilities of `device` as the compiler takes them: its features and every limit it reports, and the WGSL
/// language extensions of the `GPU` it came from (`navigator.gpu.wgslLanguageFeatures`, which the device does not
/// carry).
pub fn host_caps<A: Api>(device: &A::Device, wgsl_language_features: impl IntoIterator<Item = String>) -> HostCaps {
    let limits = device.limits();
    HostCaps {
        features: device.features(),
        limits: SUPPORTED_LIMITS.iter().filter_map(|n| limits.get(n).map(|v| (n.to_string(), v))).collect(),
        wgsl_language_features: wgsl_language_features.into_iter().collect(),
    }
}

/// Counters of the cache, for tests and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// The compiler and its cache.
pub struct Compiler {
    host: HostCaps,
    budget: usize,
    entries: RefCell<HashMap<Vec<u8>, (Outcome, usize)>>,
    order: RefCell<VecDeque<Vec<u8>>>,
    held: Cell<usize>,
    stats: Cell<CompileStats>,
}

impl Compiler {
    /// A compiler for a device with `host`'s capabilities whose cache keeps at most `budget` bytes of WGSL and refusal
    /// text (the most recent compilation is kept whatever its size).
    pub fn new(host: HostCaps, budget: usize) -> Compiler {
        Compiler {
            host,
            budget,
            entries: RefCell::new(HashMap::new()),
            order: RefCell::new(VecDeque::new()),
            held: Cell::new(0),
            stats: Cell::new(CompileStats::default()),
        }
    }

    pub fn host(&self) -> &HostCaps {
        &self.host
    }

    pub fn stats(&self) -> CompileStats {
        self.stats.get()
    }

    /// Compile `modules` with `desc`, or return the cached result of an earlier compilation of the same input.
    pub fn compile(&self, modules: &[AirModule], desc: &PipelineDesc) -> Outcome {
        let k = key(modules, desc, &self.host);
        let mut s = self.stats.get();
        if let Some((o, _)) = self.entries.borrow().get(&k) {
            s.hits += 1;
            self.stats.set(s);
            return o.clone();
        }
        s.misses += 1;
        let outcome: Outcome = Rc::new(airwgsl::compile(modules, desc, &self.host));
        let size = match &*outcome {
            Ok(c) => c.wgsl.iter().map(|w| w.source.len()).sum::<usize>(),
            Err(r) => r.reason.len() + r.construct.len(),
        };
        self.entries.borrow_mut().insert(k.clone(), (outcome.clone(), size));
        self.order.borrow_mut().push_back(k);
        self.held.set(self.held.get() + size);
        while self.held.get() > self.budget && self.order.borrow().len() > 1 {
            let old = self.order.borrow_mut().pop_front().expect("an entry");
            if let Some((_, n)) = self.entries.borrow_mut().remove(&old) {
                self.held.set(self.held.get() - n);
                s.evictions += 1;
            }
        }
        self.stats.set(s);
        outcome
    }

    /// Compile without the cache (the reference a cached result is compared with).
    // The refusal is the compiler's only failure value and is returned by value, as airwgsl::compile returns it.
    #[allow(clippy::result_large_err)]
    pub fn compile_fresh(&self, modules: &[AirModule], desc: &PipelineDesc) -> Result<Compiled, Refusal> {
        airwgsl::compile(modules, desc, &self.host)
    }

    /// The number of cached compilations.
    pub fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.borrow().is_empty()
    }
}

/// A canonical byte encoding: fixed-width little-endian integers, a tag before every optional and every variant,
/// a count before every sequence.
#[derive(Default)]
pub struct KeyWriter(pub Vec<u8>);

impl KeyWriter {
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    pub fn tag(&mut self, v: u8) {
        self.0.push(v);
    }
    pub fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.0.extend_from_slice(v);
    }
    pub fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }
    /// An `f32` by its bits: two values are the same key exactly when they have the same bits.
    pub fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    pub fn seq<T>(&mut self, items: &[T], mut f: impl FnMut(&mut Self, &T)) {
        self.u64(items.len() as u64);
        for i in items {
            f(self, i);
        }
    }
    pub fn opt<T>(&mut self, v: &Option<T>, f: impl FnOnce(&mut Self, &T)) {
        match v {
            None => self.tag(0),
            Some(x) => {
                self.tag(1);
                f(self, x);
            }
        }
    }
}

/// The cache key of compiling `modules` with `desc` on `host`.
pub fn key(modules: &[AirModule], desc: &PipelineDesc, host: &HostCaps) -> Vec<u8> {
    let mut w = KeyWriter::default();
    w.seq(modules, write_module);
    write_desc(&mut w, desc);
    let HostCaps { features, limits, wgsl_language_features } = host;
    w.u64(features.len() as u64);
    for f in features {
        w.str(f);
    }
    w.u64(limits.len() as u64);
    for (k, v) in limits {
        w.str(k);
        w.u64(*v);
    }
    w.u64(wgsl_language_features.len() as u64);
    for f in wgsl_language_features {
        w.str(f);
    }
    w.0
}

fn write_module(w: &mut KeyWriter, m: &AirModule) {
    // The registry has checked that the bitcode hashes to `hash`, so the digest stands for the bytes.
    let AirModule { hash, bitcode: _, entry } = m;
    w.0.extend_from_slice(hash);
    w.opt(entry, |w, e| {
        let EntryPoint { function, function_type, constants } = e;
        w.str(function);
        w.u64(*function_type);
        w.seq(constants, |w, c| {
            let FunctionConstant { index, name, data_type, value } = c;
            w.u64(*index);
            w.opt(name, |w, n| w.str(n));
            w.u64(*data_type);
            w.bytes(value);
        });
    });
}

/// The canonical encoding of `desc`.
pub fn write_desc(w: &mut KeyWriter, desc: &PipelineDesc) {
    let PipelineDesc { kind, bindings } = desc;
    match kind {
        PipelineKind::Render(r) => {
            w.tag(0);
            write_render(w, r);
        }
        PipelineKind::Compute(c) => {
            w.tag(1);
            let ComputeState { threads_per_threadgroup, non_uniform_grid, threadgroup_memory_lengths, stage_input } = c;
            for t in threads_per_threadgroup {
                w.u64(*t);
            }
            w.bool(*non_uniform_grid);
            w.seq(threadgroup_memory_lengths, |w, l| w.u64(*l));
            w.opt(stage_input, |w, s| {
                let StageInputDescriptor { layouts, attributes, index_type, index_buffer_index } = s;
                w.seq(layouts, |w, (a, b, c)| {
                    w.u64(*a);
                    w.u64(*b);
                    w.u64(*c);
                });
                w.seq(attributes, |w, (a, b, c)| {
                    w.u64(*a);
                    w.u64(*b);
                    w.u64(*c);
                });
                w.u64(*index_type);
                w.u64(*index_buffer_index);
            });
        }
        PipelineKind::Tile(t) => {
            w.tag(2);
            let TileState { color_attachment_pixel_formats, raster_sample_count, threadgroup_size_matches_tile_size } =
                t;
            w.seq(color_attachment_pixel_formats, |w, f| w.u64(*f));
            w.u64(*raster_sample_count);
            w.bool(*threadgroup_size_matches_tile_size);
        }
    }
    w.seq(bindings, write_bindings);
}

fn write_render(w: &mut KeyWriter, r: &RenderState) {
    let RenderState {
        vertex_descriptor,
        color_attachments,
        depth_attachment_pixel_format,
        stencil_attachment_pixel_format,
        raster_sample_count,
        alpha_to_coverage_enabled,
        alpha_to_one_enabled,
        rasterization_enabled,
        input_primitive_topology,
        primitive_type,
        tessellation,
        max_vertex_amplification_count,
    } = r;
    w.opt(vertex_descriptor, |w, v| {
        let VertexDescriptor { layouts, attributes } = v;
        w.seq(layouts, |w, l| {
            let VertexBufferLayout { stride, step_function, step_rate } = l;
            w.u64(*stride);
            w.u64(*step_function);
            w.u64(*step_rate);
        });
        w.seq(attributes, |w, a| {
            let VertexAttribute { format, offset, buffer_index } = a;
            w.u64(*format);
            w.u64(*offset);
            w.u64(*buffer_index);
        });
    });
    w.seq(color_attachments, |w, c| {
        let ColorAttachment {
            pixel_format,
            blending_enabled,
            rgb_blend_operation,
            alpha_blend_operation,
            source_rgb_blend_factor,
            destination_rgb_blend_factor,
            source_alpha_blend_factor,
            destination_alpha_blend_factor,
            write_mask,
        } = c;
        w.u64(*pixel_format);
        w.bool(*blending_enabled);
        w.u64(*rgb_blend_operation);
        w.u64(*alpha_blend_operation);
        w.u64(*source_rgb_blend_factor);
        w.u64(*destination_rgb_blend_factor);
        w.u64(*source_alpha_blend_factor);
        w.u64(*destination_alpha_blend_factor);
        w.u64(*write_mask);
    });
    w.u64(*depth_attachment_pixel_format);
    w.u64(*stencil_attachment_pixel_format);
    w.u64(*raster_sample_count);
    w.bool(*alpha_to_coverage_enabled);
    w.bool(*alpha_to_one_enabled);
    w.bool(*rasterization_enabled);
    w.u64(*input_primitive_topology);
    w.opt(primitive_type, |w, p| w.u64(*p));
    w.opt(tessellation, |w, t| {
        let Tessellation {
            partition_mode,
            max_tessellation_factor,
            factor_scale_enabled,
            factor_format,
            control_point_index_type,
            factor_step_function,
            output_winding_order,
        } = t;
        w.u64(*partition_mode);
        w.u64(*max_tessellation_factor);
        w.bool(*factor_scale_enabled);
        w.u64(*factor_format);
        w.u64(*control_point_index_type);
        w.u64(*factor_step_function);
        w.u64(*output_winding_order);
    });
    w.u64(*max_vertex_amplification_count);
}

fn write_buffer_slot(w: &mut KeyWriter, b: &BufferSlot) {
    let BufferSlot { index, alias_group, also_vertex_or_index_buffer } = b;
    w.u64(*index);
    w.opt(alias_group, |w, g| w.u32(*g));
    w.bool(*also_vertex_or_index_buffer);
}

fn write_texture_slot(w: &mut KeyWriter, t: &TextureSlot) {
    let TextureSlot { index, texture_type, pixel_format, sample_count, swizzle, mipmap_level_count } = t;
    w.u64(*index);
    w.u64(*texture_type);
    w.u64(*pixel_format);
    w.u64(*sample_count);
    w.0.extend_from_slice(swizzle);
    w.u64(*mipmap_level_count);
}

fn write_sampler_slot(w: &mut KeyWriter, s: &SamplerSlot) {
    let SamplerSlot {
        index,
        min_filter,
        mag_filter,
        mip_filter,
        address_modes,
        border_color,
        normalized_coordinates,
        compare_function,
        max_anisotropy,
        lod_min_clamp,
        lod_max_clamp,
        lod_average,
    } = s;
    w.u64(*index);
    w.u64(*min_filter);
    w.u64(*mag_filter);
    w.u64(*mip_filter);
    for m in address_modes {
        w.u64(*m);
    }
    w.u64(*border_color);
    w.bool(*normalized_coordinates);
    w.u64(*compare_function);
    w.u64(*max_anisotropy);
    w.f32(*lod_min_clamp);
    w.f32(*lod_max_clamp);
    w.bool(*lod_average);
}

fn write_bindings(w: &mut KeyWriter, b: &StageBindings) {
    let StageBindings { stage, buffers, textures, samplers, argument_buffers } = b;
    w.tag(match stage {
        airwgsl::MetalStage::Vertex => 0,
        airwgsl::MetalStage::Fragment => 1,
        airwgsl::MetalStage::Kernel => 2,
        airwgsl::MetalStage::Tile => 3,
    });
    w.seq(buffers, write_buffer_slot);
    w.seq(textures, write_texture_slot);
    w.seq(samplers, write_sampler_slot);
    w.seq(argument_buffers, |w, a| {
        let ArgumentBufferSlot { index, textures, samplers, buffers } = a;
        w.u64(*index);
        w.seq(textures, |w, (id, t)| {
            w.u64(*id);
            write_texture_slot(w, t);
        });
        w.seq(samplers, |w, (id, s)| {
            w.u64(*id);
            write_sampler_slot(w, s);
        });
        w.seq(buffers, |w, (id, b)| {
            w.u64(*id);
            write_buffer_slot(w, b);
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn compute(threads: [u64; 3]) -> PipelineDesc {
        PipelineDesc {
            kind: PipelineKind::Compute(ComputeState {
                threads_per_threadgroup: threads,
                non_uniform_grid: false,
                threadgroup_memory_lengths: Vec::new(),
                stage_input: None,
            }),
            bindings: Vec::new(),
        }
    }

    fn module(hash: u8, function: &str) -> AirModule {
        AirModule {
            hash: [hash; 32],
            bitcode: Arc::from(&b"not bitcode"[..]),
            entry: Some(EntryPoint { function: function.into(), function_type: 3, constants: Vec::new() }),
        }
    }

    #[test]
    fn keys_tell_inputs_apart() {
        let host = HostCaps::default();
        let a = key(&[module(1, "f")], &compute([1, 1, 1]), &host);
        assert_eq!(a, key(&[module(1, "f")], &compute([1, 1, 1]), &host));
        assert_ne!(a, key(&[module(2, "f")], &compute([1, 1, 1]), &host));
        assert_ne!(a, key(&[module(1, "g")], &compute([1, 1, 1]), &host));
        assert_ne!(a, key(&[module(1, "f")], &compute([2, 1, 1]), &host));
        let mut more = host.clone();
        more.features.insert("shader-f16".into());
        assert_ne!(a, key(&[module(1, "f")], &compute([1, 1, 1]), &more));
        // Adjacent strings cannot run together.
        let mut m = module(1, "fa");
        m.entry.as_mut().unwrap().constants.push(FunctionConstant {
            index: 0,
            name: Some("b".into()),
            data_type: 33,
            value: vec![1, 0, 0, 0],
        });
        let mut n = module(1, "f");
        n.entry.as_mut().unwrap().constants.push(FunctionConstant {
            index: 0,
            name: Some("ab".into()),
            data_type: 33,
            value: vec![1, 0, 0, 0],
        });
        assert_ne!(key(&[m], &compute([1, 1, 1]), &host), key(&[n], &compute([1, 1, 1]), &host));
    }

    #[test]
    fn the_cache_returns_the_first_result_and_keeps_its_budget() {
        let c = Compiler::new(HostCaps::default(), 0);
        let first = c.compile(&[module(1, "f")], &compute([1, 1, 1]));
        let again = c.compile(&[module(1, "f")], &compute([1, 1, 1]));
        assert!(Rc::ptr_eq(&first, &again));
        assert_eq!(*first, c.compile_fresh(&[module(1, "f")], &compute([1, 1, 1])));
        assert_eq!(c.stats(), CompileStats { hits: 1, misses: 1, evictions: 0 });
        c.compile(&[module(2, "f")], &compute([1, 1, 1]));
        assert_eq!(c.len(), 1, "a zero budget keeps only the newest entry");
        assert_eq!(c.stats().evictions, 1);
    }
}
