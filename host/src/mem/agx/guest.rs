//! Guest memory in the A10X layout: the copies of a texture layout object (`-[MTLTextureLayout
//! copyFromLinearBytes:...]` and `copyFromTextureMemory:...`), which move texels between two blocks of the calling
//! task's memory.
//!
//! A copy moves only the bytes the driver's copy moves: the texture memory and the linear bytes the region touches are
//! read from the task, the copy is made on the host, and only the bytes it changed are written back
//! ([`Embedder::write_guest_changed`]), so the task's other bytes are left as they are even if another of its threads
//! writes them meanwhile.

use crate::embed::{Embedder, TaskId};
use webmetal_wire::{TextureDescriptor, TextureLayoutCopyFromLinearBytes, TextureLayoutCopyFromTextureMemory};

use super::gen2::{self, DescriptorFields, SystemValues, TextureGen2};
use super::region::{self, CopyError, Linear, Outcome, Region, RegionCopy};
use crate::mem::error::MemError;

const COPY_FROM_LINEAR: &str = "-[MTLTextureLayout copyFromLinearBytes:linearOffset:linearBytesPerRow:\
                                linearBytesPerImage:toTextureMemory:textureSlice:textureLevel:textureRegion:]";
const COPY_FROM_TEXTURE: &str = "-[MTLTextureLayout copyFromTextureMemory:textureSlice:textureLevel:textureRegion:\
                                 toLinearBytes:linearOffset:linearBytesPerRow:linearBytesPerImage:]";

/// The `MTLTextureDescriptorPrivate` fields the driver's layout callers read, from the wire descriptor (as the guest's
/// `LayoutDescriptorFields` takes them).
pub fn layout_fields(d: &TextureDescriptor) -> DescriptorFields {
    DescriptorFields {
        texture_type: d.texture_type,
        pixel_format: d.pixel_format,
        width: d.width,
        height: d.height,
        depth: d.depth,
        mipmap_level_count: d.mipmap_level_count,
        sample_count: d.sample_count,
        array_length: d.array_length,
        rotation: d.rotation,
        swizzle: d.swizzle_key,
        compression_mode: d.compression_mode,
        usage: d.usage,
        resource_options: d.resource_options,
        allow_gpu_optimized_contents: d.allow_gpu_optimized_contents,
        compression_footprint: d.compression_footprint,
        color_space_conversion: d.color_space_conversion_matrix,
        storage_mode: d.storage_mode,
    }
}

/// The running system's values the layout reads, as the guest reports them.
pub fn system_values(page_size: u64, iosurface_alignment: u64) -> SystemValues {
    SystemValues { vm_page_size: page_size, iosurface_alignment }
}

/// A copy error as a memory error of `call`: every one is a path the port does not follow the driver on (LOUD).
pub fn copy_error(call: &'static str, e: CopyError) -> MemError {
    match e {
        CopyError::Unported(u) => MemError::unimplemented(format!("{call}: the A10X texture layout: {}", u.0)),
        e => MemError::unimplemented(format!("{call}: {e}")),
    }
}

/// The layout object's TextureGen2: `-[AGXTextureLayout initWithDevice:descriptor:isHeapOrBufferBacked:]`, which the
/// guest made only for a valid layout.
fn layout(call: &'static str, d: &TextureDescriptor, hb: bool, system: SystemValues) -> Result<TextureGen2, MemError> {
    let t =
        gen2::texture_layout(&layout_fields(d), hb, system).map_err(|u| copy_error(call, CopyError::Unported(u)))?;
    if !t.valid {
        return Err(MemError::invalid(format!(
            "{call}: the descriptor has no A10X texture layout (newTextureLayoutWithDescriptor:isHeapOrBufferBacked: \
             answers nil for it)"
        )));
    }
    Ok(t)
}

fn region_copy(region: &webmetal_wire::Region, slice: u64, level: u64, bpr: u64, bpi: u64) -> RegionCopy {
    RegionCopy {
        region: Region::from_mtl(
            [region.origin.x, region.origin.y, region.origin.z],
            [region.size.width, region.size.height, region.size.depth],
        ),
        // writeRegion and readRegion take the slice and the level as unsigned int.
        slice: slice as u32,
        level: level as u32,
        linear: Linear { bytes_per_row: bpr, bytes_per_image: bpi },
    }
}

/// The twiddle functions return without copying when either pointer is null (`cbz x0`/`cbz x1` at their entry), the
/// linear copy and `agxaAssertBufferIsValid` read and write through it: which applies depends on the layout and the
/// region, and is not ported.
fn non_null(call: &'static str, memory: u64, linear: u64) -> Result<(), MemError> {
    if memory == 0 || linear == 0 {
        return Err(MemError::unimplemented(format!(
            "{call} with a null texture memory or linear pointer: the A10X driver skips the copy or faults depending on \
             the layout, which is not ported"
        )));
    }
    Ok(())
}

fn guest_range(call: &'static str, base: u64, offset: u64, length: u64) -> Result<(u64, usize), MemError> {
    let at = base.checked_add(offset).ok_or_else(|| MemError::invalid(format!("{call}: the address wraps")))?;
    let length = usize::try_from(length).map_err(|_| MemError::invalid(format!("{call}: {length} bytes")))?;
    at.checked_add(length as u64).ok_or_else(|| MemError::invalid(format!("{call}: the range wraps")))?;
    Ok((at, length))
}

/// `-[AGXTextureLayout copyFromLinearBytes:...]` (0x21c9c04b4): `writeRegion` into the texture memory. Answers the
/// assertion the driver logs when it copies nothing.
pub fn copy_from_linear_bytes(
    e: &dyn Embedder,
    task: TaskId,
    r: &TextureLayoutCopyFromLinearBytes,
) -> Result<Option<&'static str>, MemError> {
    const CALL: &str = COPY_FROM_LINEAR;
    let t = layout(CALL, &r.descriptor, r.heap_or_buffer_backed, system_values(r.page_size, r.iosurface_alignment))?;
    let c = region_copy(&r.region, r.slice, r.level, r.bytes_per_row, r.bytes_per_image);
    if let Some(refusal) = region::validate_region(&t, &c).map_err(|e| copy_error(CALL, e))? {
        return Ok(Some(refusal.assertion()));
    }
    non_null(CALL, r.memory.0, r.bytes.0.wrapping_add(r.offset))?;
    let extent = region::extent(&t, &c).map_err(|e| copy_error(CALL, e))?.expect("the region was validated");
    let (linear_at, linear_len) = guest_range(CALL, r.bytes.0, r.offset, extent.linear)?;
    let mut linear = vec![0u8; linear_len];
    e.read_guest(task, linear_at, &mut linear)?;
    let Some((first, end)) = extent.texture else { return Ok(None) };
    let (memory_at, memory_len) = guest_range(CALL, r.memory.0, first, end - first)?;
    let mut held = vec![0u8; memory_len];
    e.read_guest(task, memory_at, &mut held)?;
    let mut new = held.clone();
    match region::write_region(&t, &c, &mut new, first, &linear).map_err(|e| copy_error(CALL, e))? {
        Outcome::Copied => {}
        Outcome::Refused(r) => unreachable!("validated above, refused with {r:?}"),
    }
    e.write_guest_changed(task, memory_at, &new, &held)?;
    Ok(None)
}

/// `-[AGXTextureLayout copyFromTextureMemory:...]` (0x21c9c0564): `readRegion` from the texture memory. Answers the
/// assertion the driver logs when it copies nothing.
pub fn copy_from_texture_memory(
    e: &dyn Embedder,
    task: TaskId,
    r: &TextureLayoutCopyFromTextureMemory,
) -> Result<Option<&'static str>, MemError> {
    const CALL: &str = COPY_FROM_TEXTURE;
    let t = layout(CALL, &r.descriptor, r.heap_or_buffer_backed, system_values(r.page_size, r.iosurface_alignment))?;
    let c = region_copy(&r.region, r.slice, r.level, r.bytes_per_row, r.bytes_per_image);
    if let Some(refusal) = region::validate_region(&t, &c).map_err(|e| copy_error(CALL, e))? {
        return Ok(Some(refusal.assertion()));
    }
    non_null(CALL, r.memory.0, r.bytes.0.wrapping_add(r.offset))?;
    let extent = region::extent(&t, &c).map_err(|e| copy_error(CALL, e))?.expect("the region was validated");
    let (linear_at, linear_len) = guest_range(CALL, r.bytes.0, r.offset, extent.linear)?;
    let mut held = vec![0u8; linear_len];
    e.read_guest(task, linear_at, &mut held)?;
    let Some((first, end)) = extent.texture else { return Ok(None) };
    let (memory_at, memory_len) = guest_range(CALL, r.memory.0, first, end - first)?;
    let mut memory = vec![0u8; memory_len];
    e.read_guest(task, memory_at, &mut memory)?;
    let mut new = held.clone();
    match region::read_region(&t, &c, &memory, first, &mut new).map_err(|e| copy_error(CALL, e))? {
        Outcome::Copied => {}
        Outcome::Refused(r) => unreachable!("validated above, refused with {r:?}"),
    }
    e.write_guest_changed(task, linear_at, &new, &held)?;
    Ok(None)
}
