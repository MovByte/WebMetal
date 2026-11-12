//! IOSurface-backed textures (`newTextureWithDescriptor:iosurface:plane:`) and the table of the planes they use.
//!
//! A surface is a kernel object every task can reach; a texture over one of its planes uses the plane's memory as its
//! storage. The plane's bytes live in the embedder's IOSurface implementation ([`Embedder::read_surface`],
//! [`Embedder::write_surface_changed`]); the CPU reaches them between `IOSurfaceLock` and `IOSurfaceUnlock`, after
//! which the surface's seed has changed (plan section 6.2, private track K2). The table keeps one [`SurfacePlane`] per
//! (surface, plane), shared by every task and every texture over it, with a shadow of the plane's bytes:
//!
//! - when a command buffer that uses such a texture is committed, a changed seed makes the executor read the plane,
//!   compare it with the shadow and write the rows that changed into the texture ([`Memory::prepare_surface_texture`]);
//! - after GPU work that wrote the texture, its contents are read back, and the rows that differ from the shadow are
//!   written into the surface ([`Embedder::write_surface_changed`]) before [`Embedder::surface_note_gpu_write`]
//!   tells the kernel to change the seed ([`Memory::read_back_surface_texture`]).
//!
//! The exactness limit of plan section 9 (row 6) applies: CPU access the kernel does not see, outside lock and unlock,
//! is not tracked.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::embed::{Embedder, SurfacePlane as PlaneDesc, TaskId};
use crate::gpu::{Api, GpuFuture};
use webmetal_wire::{CreateTextureWithIOSurface, SurfaceId};

use super::calls::Calls;
use super::error::MemError;
use super::layout::{RegionArgs, RegionLayout};
use super::memory::Memory;
use super::texture::{Texture, TextureBytes, TextureRead};

/// One plane of a surface, as the textures over it share it.
pub struct SurfacePlane {
    pub id: SurfaceId,
    pub plane: u32,
    pub geometry: PlaneDesc,
    /// The plane's bytes as the executor last saw them, `bytes_per_row` × `height`.
    shadow: RefCell<Vec<u8>>,
    /// The seed the shadow was read at; `None` before the first read and after a GPU write.
    seed: Cell<Option<u32>>,
    /// Advances whenever the shadow changes.
    version: Cell<u64>,
    /// The rows (first, past last) of the latest change, for textures one version behind.
    changed: Cell<(u32, u32)>,
}

impl SurfacePlane {
    pub fn version(&self) -> u64 {
        self.version.get()
    }
}

/// A texture's contents read back after GPU work wrote it, to write into its surface.
pub struct SurfaceReadback<A: Api> {
    texture: Rc<Texture<A>>,
    read: TextureRead<A>,
}

impl<A: Api> Memory<A> {
    /// `newTextureWithDescriptor:iosurface:plane:` (CreateTextureWithIOSurface).
    pub fn create_texture_iosurface(
        &self,
        g: &impl Calls<A>,
        e: &dyn Embedder,
        task: TaskId,
        r: &CreateTextureWithIOSurface,
    ) -> Result<Rc<Texture<A>>, MemError> {
        const CALL: &str = "-[MTLDevice newTextureWithDescriptor:iosurface:plane:]";
        let desc = e.surface_lookup(r.iosurface)?;
        let plane = u32::try_from(r.plane).map_err(|_| MemError::undefined(CALL, "plane"))?;
        let geometry = if desc.planes.is_empty() {
            if plane != 0 {
                return Err(MemError::undefined(CALL, format!("plane {plane} of a surface without planes")));
            }
            PlaneDesc {
                width: desc.width,
                height: desc.height,
                bytes_per_row: desc.bytes_per_row,
                bytes_per_element: desc.bytes_per_element,
                element_width: desc.element_width,
                element_height: desc.element_height,
                offset: 0,
                size: desc.alloc_size,
            }
        } else {
            *desc
                .planes
                .get(plane as usize)
                .ok_or_else(|| MemError::undefined(CALL, format!("plane {plane} of {}", desc.planes.len())))?
        };
        let d = &r.descriptor;
        if d.texture_type != 2 || d.mipmap_level_count != 1 || d.sample_count != 1 || d.array_length != 1 {
            return Err(MemError::undefined(CALL, "an IOSurface texture is a 2D texture of one level and one sample"));
        }
        if d.width > geometry.width || d.height > geometry.height {
            return Err(MemError::undefined(
                CALL,
                format!("a {}x{} texture over a {}x{} plane", d.width, d.height, geometry.width, geometry.height),
            ));
        }
        let key = (r.iosurface, plane);
        let shared = self.surfaces.borrow().get(&key).cloned();
        let entry = match shared {
            Some(p) if p.geometry == geometry => p,
            _ => {
                let p = Rc::new(SurfacePlane {
                    id: r.iosurface,
                    plane,
                    geometry,
                    shadow: RefCell::new(vec![0; (geometry.bytes_per_row * geometry.height) as usize]),
                    seed: Cell::new(None),
                    version: Cell::new(0),
                    changed: Cell::new((0, 0)),
                });
                self.surfaces.borrow_mut().insert(key, p.clone());
                p
            }
        };
        let t = self.build_texture(g, CALL, (task, r.texture), d, Some(entry))?;
        let Some(b) = t.blocks.filter(|b| !b.compressed()) else {
            return Err(MemError::unimplemented(format!("{CALL}: a compressed or depth-stencil IOSurface texture")));
        };
        if u64::from(b.bytes) * geometry.element_width != geometry.bytes_per_element || geometry.element_height != 1 {
            return Err(MemError::undefined(
                CALL,
                format!(
                    "{}-byte texels over a plane of {}-byte elements {} pixels wide",
                    b.bytes, geometry.bytes_per_element, geometry.element_width
                ),
            ));
        }
        if d.width * u64::from(b.bytes) > geometry.bytes_per_row {
            return Err(MemError::undefined(CALL, "the texture's rows are longer than the plane's"));
        }
        t.set_surface_state(u64::MAX, false);
        Ok(t)
    }

    /// A layout of rows `first..last` of the texture, as tight rows.
    fn surface_rows(&self, t: &Texture<A>, first: u32, last: u32) -> Result<RegionLayout, MemError> {
        let b = t.blocks.expect("IOSurface textures are uncompressed");
        let a = RegionArgs {
            origin: (0, u64::from(first), 0),
            size: (u64::from(t.geometry.width), u64::from(last - first), 1),
            level: 0,
            slice: 0,
            bytes_per_row: u64::from(t.geometry.width) * u64::from(b.bytes),
            bytes_per_image: 0,
        };
        RegionLayout::new("IOSurface rows", &t.geometry, b, &a)
    }

    /// Before GPU work uses an IOSurface texture (at the commit of a command buffer that names it): if the surface's
    /// seed changed, read the plane and keep the rows that changed; then write into the texture every row it does not
    /// hold yet.
    pub fn prepare_surface_texture(&self, g: &impl Calls<A>, e: &dyn Embedder, t: &Texture<A>) -> Result<(), MemError> {
        const CALL: &str = "IOSurface texture use";
        let p = t.surface.as_ref().expect("an IOSurface texture");
        let desc = e.surface_lookup(p.id)?;
        let bpr = p.geometry.bytes_per_row as usize;
        if p.seed.get() != Some(desc.seed) {
            let mut bytes = vec![0u8; bpr * p.geometry.height as usize];
            e.read_surface(p.id, p.geometry.offset, &mut bytes)?;
            let mut shadow = p.shadow.borrow_mut();
            let differs = |y: usize| bytes[y * bpr..(y + 1) * bpr] != shadow[y * bpr..(y + 1) * bpr];
            let rows = p.geometry.height as usize;
            if let Some(first) = (0..rows).find(|&y| differs(y)) {
                let last = (0..rows).rev().find(|&y| differs(y)).expect("a row differs") + 1;
                *shadow = bytes;
                p.version.set(p.version.get() + 1);
                p.changed.set((first as u32, last as u32));
            }
            p.seed.set(Some(desc.seed));
        }
        let (held, written) = t.surface_state();
        if held == p.version() {
            return Ok(());
        }
        let (first, last) = if held != u64::MAX && held + 1 == p.version() { p.changed.get() } else { (0, u32::MAX) };
        let (first, last) = (first.min(t.geometry.height), last.min(t.geometry.height));
        if first < last {
            let l = self.surface_rows(t, first, last)?;
            let shadow = p.shadow.borrow();
            let mut tight = Vec::with_capacity(l.tight_len() as usize);
            for y in first as usize..last as usize {
                tight.extend_from_slice(&shadow[y * bpr..y * bpr + l.row_bytes as usize]);
            }
            self.upload_region(g, CALL, t, &l, (0, 0), &tight)?;
        }
        t.set_surface_state(p.version(), written);
        Ok(())
    }

    /// After GPU work that wrote an IOSurface texture: encode the read of its contents into `encoder`.
    pub fn read_back_surface_texture(
        &self,
        g: &impl Calls<A>,
        encoder: &A::CommandEncoder,
        t: &Rc<Texture<A>>,
    ) -> Result<Option<SurfaceReadback<A>>, MemError> {
        let (held, written) = t.surface_state();
        if !written {
            return Ok(None);
        }
        let l = self.surface_rows(t, 0, t.geometry.height)?;
        let read = self.encode_read(g, "IOSurface write-back", encoder, t, &l, (0, 0))?;
        t.set_surface_state(held, false);
        Ok(Some(SurfaceReadback { texture: t.clone(), read }))
    }
}

/// An IOSurface texture's contents, read back.
pub struct SurfaceArrived<A: Api> {
    texture: Rc<Texture<A>>,
    bytes: TextureBytes<A>,
}

impl<A: Api> SurfaceReadback<A> {
    /// Map the read once the GPU has run it. Makes no WebGPU call that can raise a device-timeline error.
    pub fn wait(self) -> GpuFuture<Result<SurfaceArrived<A>, MemError>> {
        Box::pin(async move {
            let bytes = self.read.wait().await?;
            Ok(SurfaceArrived { texture: self.texture, bytes })
        })
    }
}

impl<A: Api> SurfaceArrived<A> {
    /// Write the rows that differ from the plane's shadow into the surface and tell the kernel the GPU wrote it.
    pub fn merge(self, mem: &Memory<A>, e: &dyn Embedder) -> Result<(), MemError> {
        let t = self.texture;
        let p = t.surface.as_ref().expect("an IOSurface texture").clone();
        let tight = self.bytes.release(mem);
        let bpr = p.geometry.bytes_per_row as usize;
        let row = tight.len() / t.geometry.height as usize;
        let mut changed = None;
        {
            let mut shadow = p.shadow.borrow_mut();
            for y in 0..t.geometry.height as usize {
                let new = &tight[y * row..(y + 1) * row];
                let old = &mut shadow[y * bpr..y * bpr + row];
                if new != old {
                    e.write_surface_changed(p.id, p.geometry.offset + (y * bpr) as u64, new, old)?;
                    old.copy_from_slice(new);
                    let (a, _) = changed.unwrap_or((y, y));
                    changed = Some((a, y + 1));
                }
            }
        }
        if let Some((first, last)) = changed {
            p.version.set(p.version.get() + 1);
            p.changed.set((first as u32, last as u32));
            e.surface_note_gpu_write(p.id, p.plane);
            p.seed.set(None);
        }
        t.set_surface_state(p.version(), false);
        Ok(())
    }
}
