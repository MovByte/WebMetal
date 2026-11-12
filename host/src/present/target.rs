//! Presentation targets: what a display's frames are written into.
//!
//! In a page the target is the display's canvas (`GPUCanvasContext`, `super::web`). Outside a browser it is an
//! [`Offscreen`] target, which performs the canvas's content-timeline steps of WebGPU §21.2 itself over a texture
//! of the device, so that what a canvas would show can be read back and compared.

use std::cell::RefCell;
use std::rc::Rc;

use crate::gpu::idl::{
    BufferDescriptor, BufferUsage, CanvasAlphaMode, CanvasConfiguration, CanvasToneMappingMode,
    CommandBufferDescriptor, CommandEncoderDescriptor, ErrorFilter, Extent3D, Extent3DDict, MapMode, Origin3D,
    Origin3DDict, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureDescriptor, TextureFormat,
    TextureUsage,
};
use crate::gpu::tables::format_required_feature;
use crate::gpu::{
    Api, Buffer as _, CommandEncoder as _, Device as _, Exception, ExceptionKind, GpuFuture, MappedRange as _,
    Queue as _, Size64, Texture as _,
};

use super::canvas::SUPPORTED_CONTEXT_FORMATS;

/// What a display's frames are written into: a canvas context, or what stands in for one.
pub trait Target<A: Api> {
    /// `canvas.width` and `canvas.height`.
    fn size(&self) -> (u32, u32);

    /// Set `canvas.width` and `canvas.height` ("update the canvas size", WebGPU §21.4.2).
    fn set_size(&mut self, width: u32, height: u32);

    /// `GPUCanvasContext.configure(configuration)`.
    fn configure(&mut self, configuration: &CanvasConfiguration<'_, A>) -> Result<(), Exception>;

    /// `GPUCanvasContext.unconfigure()`.
    fn unconfigure(&mut self);

    /// `getCurrentTexture()`, then `GPUQueue.writeTexture()` of `data` (rows of `bytes_per_row` bytes) into its
    /// `width` × `height` texels at the origin.
    fn write_frame(
        &mut self,
        queue: &A::Queue,
        data: &[u8],
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) -> Result<(), Exception>;

    /// The frame is complete. A canvas is presented when the browser updates the rendering, so this does nothing for
    /// one; a target that presents on request presents here.
    fn present(&mut self) -> Result<(), Exception>;
}

/// The configuration of an [`Offscreen`] target: `GPUCanvasConfiguration` with its device.
#[derive(Clone, Debug)]
pub struct OffscreenConfiguration<A: Api> {
    pub device: A::Device,
    pub format: TextureFormat,
    pub usage: u32,
    pub view_formats: Vec<TextureFormat>,
    pub color_space: String,
    pub tone_mapping: CanvasToneMappingMode,
    pub alpha_mode: CanvasAlphaMode,
}

struct State<A: Api> {
    width: u32,
    height: u32,
    /// `[[configuration]]`.
    configuration: Option<OffscreenConfiguration<A>>,
    /// `[[textureDescriptor]]`.
    descriptor: Option<TextureDescriptor>,
    /// `[[drawingBuffer]]`: `None` is a transparent black image not yet given storage.
    drawing_buffer: Option<A::Texture>,
    /// `[[currentTexture]]`.
    current: Option<A::Texture>,
    /// Times the rendering was updated ([`Target::present`]).
    presented: u64,
}

/// A presentation target that is no canvas: the drawing buffer is a texture of the configured device.
///
/// It performs the content-timeline steps of `configure()`, `unconfigure()` and `getCurrentTexture()`, "replace the
/// drawing buffer", "expire the current texture" and "get a copy of the image contents of a context" (WebGPU §21.2),
/// and [`Target::present`] is "updating the rendering of a WebGPU canvas". Two things differ from a canvas, neither of
/// which changes what is shown: the drawing buffer's texture is the current texture's storage and also carries
/// `COPY_SRC`, the read the compositor makes; and expiring the current texture stops handing it out without
/// destroying it, since it is that storage.
///
/// Cloning refers to the same target.
pub struct Offscreen<A: Api> {
    state: Rc<RefCell<State<A>>>,
}

impl<A: Api> Clone for Offscreen<A> {
    fn clone(&self) -> Offscreen<A> {
        Offscreen { state: self.state.clone() }
    }
}

impl<A: Api> std::fmt::Debug for Offscreen<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.state.borrow();
        let format = s.configuration.as_ref().map(|c| c.format.as_str());
        write!(f, "Offscreen({}x{}, {format:?}, presented {})", s.width, s.height, s.presented)
    }
}

/// A copy of the image contents of a target, as the browser would show it or hand it to another API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub width: u32,
    pub height: u32,
    /// `None` for an unconfigured target, whose image is transparent black tagged "srgb".
    pub format: Option<TextureFormat>,
    pub color_space: String,
    pub tone_mapping: Option<CanvasToneMappingMode>,
    /// Tagged as opaque: the alpha channel was cleared to 1.0.
    pub opaque: bool,
    /// Rows of `width` texels, no padding.
    pub bytes: Vec<u8>,
}

/// Bytes of one texel of a supported context format or of an sRGB view format of one.
pub fn texel_bytes(format: TextureFormat) -> u32 {
    match format {
        TextureFormat::Bgra8unorm
        | TextureFormat::Bgra8unormSrgb
        | TextureFormat::Rgba8unorm
        | TextureFormat::Rgba8unormSrgb => 4,
        TextureFormat::Rgba16float => 8,
        _ => unreachable!("not a supported context format"),
    }
}

impl<A: Api> Offscreen<A> {
    /// A target of `width` × `height`, not configured.
    pub fn new(width: u32, height: u32) -> Offscreen<A> {
        Offscreen {
            state: Rc::new(RefCell::new(State {
                width,
                height,
                configuration: None,
                descriptor: None,
                drawing_buffer: None,
                current: None,
                presented: 0,
            })),
        }
    }

    /// `getConfiguration()`.
    pub fn configuration(&self) -> Option<OffscreenConfiguration<A>> {
        self.state.borrow().configuration.clone()
    }

    /// Times the rendering was updated.
    pub fn presented(&self) -> u64 {
        self.state.borrow().presented
    }

    /// `getCurrentTexture()`.
    pub fn get_current_texture(&self) -> Result<A::Texture, Exception> {
        let mut s = self.state.borrow_mut();
        let Some(c) = s.configuration.clone() else {
            return Err(Exception::new(
                ExceptionKind::InvalidStateError,
                "getCurrentTexture() on an unconfigured context",
            ));
        };
        if let Some(t) = &s.current {
            return Ok(t.clone());
        }
        s.replace_drawing_buffer();
        let mut d = s.descriptor.clone().expect("a configured context has a texture descriptor");
        d.usage |= TextureUsage::COPY_SRC;
        let t = c.device.create_texture(&d)?;
        s.drawing_buffer = Some(t.clone());
        s.current = Some(t.clone());
        Ok(t)
    }

    /// The drawing buffer's bytes, unchanged: rows of `width` texels. `None` when the target is not configured.
    pub fn drawing_buffer(&self) -> GpuFuture<Result<Option<Vec<u8>>, String>> {
        let s = self.state.borrow();
        let Some(c) = s.configuration.clone() else { return Box::pin(async { Ok(None) }) };
        let (w, h) = (s.width, s.height);
        let n = (w as usize) * (h as usize) * texel_bytes(c.format) as usize;
        let Some(t) = s.drawing_buffer.clone() else { return Box::pin(async move { Ok(Some(vec![0; n])) }) };
        let read = read_texture::<A>(&c.device, &t, c.format, (0, 0), w, h);
        Box::pin(async move { read.await.map(Some) })
    }

    /// "Get a copy of the image contents of a context" (WebGPU §21.2): the drawing buffer, with the alpha channel
    /// cleared to 1.0 when the alpha mode is "opaque". Reading it waits for the work submitted before it.
    pub fn snapshot(&self) -> GpuFuture<Result<Snapshot, String>> {
        let (w, h, c) = {
            let s = self.state.borrow();
            (s.width, s.height, s.configuration.clone())
        };
        let bytes = self.drawing_buffer();
        Box::pin(async move {
            let Some(c) = c else {
                return Ok(Snapshot {
                    width: w,
                    height: h,
                    format: None,
                    color_space: "srgb".into(),
                    tone_mapping: None,
                    opaque: false,
                    bytes: vec![0; w as usize * h as usize * 4],
                });
            };
            let mut bytes = bytes.await?.expect("configured");
            let opaque = c.alpha_mode == CanvasAlphaMode::Opaque;
            if opaque {
                clear_alpha(c.format, &mut bytes);
            }
            Ok(Snapshot {
                width: w,
                height: h,
                format: Some(c.format),
                color_space: c.color_space,
                tone_mapping: Some(c.tone_mapping),
                opaque,
                bytes,
            })
        })
    }
}

/// Set every texel's alpha to 1.0 in `format`.
fn clear_alpha(format: TextureFormat, bytes: &mut [u8]) {
    match format {
        TextureFormat::Bgra8unorm | TextureFormat::Rgba8unorm => {
            for t in bytes.as_chunks_mut::<4>().0 {
                t[3] = 0xff;
            }
        }
        TextureFormat::Rgba16float => {
            // 1.0 in binary16 is 0x3c00.
            for t in bytes.as_chunks_mut::<8>().0 {
                t[6..8].copy_from_slice(&0x3c00u16.to_le_bytes());
            }
        }
        _ => unreachable!("not a supported context format"),
    }
}

impl<A: Api> State<A> {
    /// "Expire the current texture". The texture is the drawing buffer's storage, so it is not destroyed.
    fn expire_current_texture(&mut self) {
        self.current = None;
    }

    /// "Replace the drawing buffer": a transparent black image of the canvas's size, in the configured format.
    fn replace_drawing_buffer(&mut self) {
        self.expire_current_texture();
        self.drawing_buffer = None;
    }

    /// "The GPUTextureDescriptor for the canvas and configuration".
    fn descriptor_for(&self, c: &OffscreenConfiguration<A>) -> TextureDescriptor {
        let mut d = TextureDescriptor::new(
            Extent3D::Extent3DDict(Extent3DDict { width: self.width, height: self.height, depth_or_array_layers: 1 }),
            c.format,
            c.usage,
        );
        d.view_formats = c.view_formats.clone();
        d.label = "offscreen drawing buffer".into();
        d
    }
}

/// "Validate texture format required features" (WebGPU #abstract-opdef-validate-texture-format-required-features):
/// `TypeError` when the device lacks the feature `format` requires.
fn check_required_features<A: Api>(device: &A::Device, format: TextureFormat) -> Result<(), Exception> {
    if let Some(f) = format_required_feature(format)
        && !device.features().contains(f.as_str())
    {
        return Err(Exception::new(
            ExceptionKind::TypeError,
            format!("texture format \"{}\" requires feature \"{}\"", format.as_str(), f.as_str()),
        ));
    }
    Ok(())
}

impl<A: Api> Target<A> for Offscreen<A> {
    fn size(&self) -> (u32, u32) {
        let s = self.state.borrow();
        (s.width, s.height)
    }

    fn set_size(&mut self, width: u32, height: u32) {
        let mut s = self.state.borrow_mut();
        s.width = width;
        s.height = height;
        s.replace_drawing_buffer();
        if let Some(c) = s.configuration.clone() {
            s.descriptor = Some(s.descriptor_for(&c));
        }
    }

    fn configure(&mut self, configuration: &CanvasConfiguration<'_, A>) -> Result<(), Exception> {
        let device = configuration.device;
        check_required_features::<A>(device, configuration.format)?;
        for f in &configuration.view_formats {
            check_required_features::<A>(device, *f)?;
        }
        if !SUPPORTED_CONTEXT_FORMATS.contains(&configuration.format) {
            return Err(Exception::new(
                ExceptionKind::TypeError,
                format!("\"{}\" is not a supported context format", configuration.format.as_str()),
            ));
        }
        if configuration.usage & TextureUsage::TRANSIENT_ATTACHMENT != 0 {
            return Err(Exception::new(ExceptionKind::TypeError, "a canvas texture cannot be a transient attachment"));
        }
        let c = OffscreenConfiguration {
            device: device.clone(),
            format: configuration.format,
            usage: configuration.usage,
            view_formats: configuration.view_formats.clone(),
            color_space: configuration.color_space.as_str().to_string(),
            tone_mapping: configuration.tone_mapping.mode,
            alpha_mode: configuration.alpha_mode,
        };
        let descriptor = {
            let mut s = self.state.borrow_mut();
            let d = s.descriptor_for(&c);
            s.configuration = Some(c);
            s.descriptor = Some(d.clone());
            s.replace_drawing_buffer();
            d
        };
        // Device timeline: "validating GPUTextureDescriptor(device, descriptor) must return true", else a validation
        // error. Creating a texture from the descriptor runs exactly that validation and raises the same error.
        let probe = device.create_texture(&descriptor)?;
        probe.destroy();
        Ok(())
    }

    fn unconfigure(&mut self) {
        let mut s = self.state.borrow_mut();
        s.configuration = None;
        s.descriptor = None;
        s.replace_drawing_buffer();
    }

    fn write_frame(
        &mut self,
        queue: &A::Queue,
        data: &[u8],
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) -> Result<(), Exception> {
        let t = self.get_current_texture()?;
        let mut layout = TexelCopyBufferLayout::new();
        layout.bytes_per_row = Some(bytes_per_row);
        layout.rows_per_image = Some(height);
        let size = Extent3D::Extent3DDict(Extent3DDict { width, height, depth_or_array_layers: 1 });
        queue.write_texture(&TexelCopyTextureInfo::new(&t), data, &layout, &size)
    }

    fn present(&mut self) -> Result<(), Exception> {
        let mut s = self.state.borrow_mut();
        s.expire_current_texture();
        if s.configuration.is_some() {
            s.presented += 1;
        }
        Ok(())
    }
}

/// Read the `width` × `height` texels at `origin` of level 0 of a texture in a format of [`texel_bytes`] back as tight
/// rows, under error scopes of its own.
pub fn read_texture<A: Api>(
    device: &A::Device,
    texture: &A::Texture,
    format: TextureFormat,
    origin: (u32, u32),
    width: u32,
    height: u32,
) -> GpuFuture<Result<Vec<u8>, String>> {
    let row = width * texel_bytes(format);
    // copyTextureToBuffer: bytesPerRow is a multiple of 256 (WebGPU #abstract-opdef-validating-gputexelcopybufferinfo).
    let pitch = row.next_multiple_of(256);
    let size = u64::from(pitch) * u64::from(height);
    let filters = [ErrorFilter::Validation, ErrorFilter::OutOfMemory, ErrorFilter::Internal];
    for f in filters {
        device.push_error_scope(f);
    }
    let staged = (|| -> Result<A::Buffer, Exception> {
        let mut d = BufferDescriptor::new(Size64::new(size)?, BufferUsage::COPY_DST | BufferUsage::MAP_READ);
        d.label = "offscreen readback".into();
        let buffer = device.create_buffer(&d)?;
        let encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: "offscreen readback".into() });
        let mut dst = TexelCopyBufferInfo::new(&buffer);
        dst.bytes_per_row = Some(pitch);
        dst.rows_per_image = Some(height);
        let mut src = TexelCopyTextureInfo::new(texture);
        src.origin = Origin3D::Origin3DDict(Origin3DDict { x: origin.0, y: origin.1, z: 0 });
        encoder.copy_texture_to_buffer(
            &src,
            &dst,
            &Extent3D::Extent3DDict(Extent3DDict { width, height, depth_or_array_layers: 1 }),
        )?;
        let cb = encoder.finish(&CommandBufferDescriptor { label: "offscreen readback".into() });
        device.queue().submit(&[&cb]);
        Ok(buffer)
    })();
    let pops: Vec<_> = filters.iter().map(|_| device.pop_error_scope()).collect();
    Box::pin(async move {
        let mut errors = Vec::new();
        for p in pops {
            match p.await {
                Ok(Some(e)) => errors.push(e.to_string()),
                Ok(None) => {}
                Err(e) => errors.push(format!("popErrorScope rejected: {e}")),
            }
        }
        let buffer = staged.map_err(|e| e.to_string())?;
        if !errors.is_empty() {
            return Err(errors.join("; "));
        }
        buffer.map_async(MapMode::READ, Size64::new_const(0), None).await.map_err(|e| e.to_string())?;
        let range = buffer.get_mapped_range(Size64::new_const(0), None).map_err(|e| e.to_string())?;
        let mut out = vec![0u8; row as usize * height as usize];
        for y in 0..height as usize {
            range
                .read(y as u64 * u64::from(pitch), &mut out[y * row as usize..(y + 1) * row as usize])
                .map_err(|e| e.to_string())?;
        }
        drop(range);
        buffer.unmap();
        Ok(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_clears_the_alpha_of_each_texel() {
        let mut b = vec![1, 2, 3, 4, 5, 6, 7, 8];
        clear_alpha(TextureFormat::Bgra8unorm, &mut b);
        assert_eq!(b, vec![1, 2, 3, 0xff, 5, 6, 7, 0xff]);
        let mut h = vec![0x11; 16];
        clear_alpha(TextureFormat::Rgba16float, &mut h);
        assert_eq!(&h[6..8], &[0x00, 0x3c]);
        assert_eq!(&h[14..16], &[0x00, 0x3c]);
        assert_eq!(&h[0..6], &[0x11; 6]);
    }
}
