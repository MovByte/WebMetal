//! The presentation target over a page's `GPUCanvasContext` (feature `web`, wasm32).
//!
//! The canvas is the display: its drawing buffer is shown when the browser updates the rendering (WebGPU §21.3), so
//! [`Target::present`] has nothing to do. Every call goes to the JavaScript object by its IDL name, as the binding in
//! `crate::gpu::web` does.

use js_sys::{Array, Function, Object, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};

use crate::gpu::idl::{CanvasConfiguration, CanvasToneMappingMode};
use crate::gpu::web::Web;
use crate::gpu::{Exception, ExceptionKind};

use super::target::Target;

fn get(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

fn set(obj: &JsValue, key: &str, value: &JsValue) -> Result<(), Exception> {
    Reflect::set(obj, &JsValue::from_str(key), value).map(|_| ()).map_err(exception)
}

/// A JavaScript exception as an [`Exception`].
fn exception(e: JsValue) -> Exception {
    let name = get(&e, "name").as_string().unwrap_or_default();
    let message = get(&e, "message").as_string().unwrap_or_else(|| format!("{e:?}"));
    let kind = match name.as_str() {
        "TypeError" => ExceptionKind::TypeError,
        "RangeError" => ExceptionKind::RangeError,
        "OperationError" => ExceptionKind::OperationError,
        "AbortError" => ExceptionKind::AbortError,
        "SecurityError" => ExceptionKind::SecurityError,
        "InvalidStateError" => ExceptionKind::InvalidStateError,
        _ => ExceptionKind::Other(name),
    };
    Exception::new(kind, message)
}

/// `obj.method(...args)`.
fn call(obj: &JsValue, iface: &str, method: &str, args: &[JsValue]) -> Result<JsValue, Exception> {
    let f = get(obj, method);
    let Some(f) = f.dyn_ref::<Function>() else {
        return Err(Exception::unsupported(format!("the browser has no {iface}.{method}()")));
    };
    let arr = Array::new();
    for a in args {
        arr.push(a);
    }
    Reflect::apply(f, obj, &arr).map_err(exception)
}

fn dict(entries: &[(&str, JsValue)]) -> Result<JsValue, Exception> {
    let o: JsValue = Object::new().into();
    for (k, v) in entries {
        set(&o, k, v)?;
    }
    Ok(o)
}

/// A page's `GPUCanvasContext` as a presentation target.
pub struct CanvasTarget {
    context: JsValue,
    canvas: JsValue,
}

impl CanvasTarget {
    /// The target over `context`, the canvas's `getContext("webgpu")`.
    pub fn new(context: JsValue) -> Result<CanvasTarget, Exception> {
        let canvas = get(&context, "canvas");
        if canvas.is_undefined() || canvas.is_null() {
            return Err(Exception::new(ExceptionKind::TypeError, "the value is not a GPUCanvasContext"));
        }
        Ok(CanvasTarget { context, canvas })
    }

    /// The `GPUCanvasContext`.
    pub fn context(&self) -> &JsValue {
        &self.context
    }
}

impl Target<Web> for CanvasTarget {
    fn size(&self) -> (u32, u32) {
        let n = |k| get(&self.canvas, k).as_f64().unwrap_or(0.0) as u32;
        (n("width"), n("height"))
    }

    fn set_size(&mut self, width: u32, height: u32) {
        // Setting the attributes runs "update the canvas size" (WebGPU §21.4.2); it does not throw for these values.
        let _ = set(&self.canvas, "width", &JsValue::from_f64(f64::from(width)));
        let _ = set(&self.canvas, "height", &JsValue::from_f64(f64::from(height)));
    }

    fn configure(&mut self, c: &CanvasConfiguration<'_, Web>) -> Result<(), Exception> {
        let view_formats = Array::new();
        for f in &c.view_formats {
            view_formats.push(&JsValue::from_str(f.as_str()));
        }
        let tone_mapping = dict(&[("mode", JsValue::from_str(c.tone_mapping.mode.as_str()))])?;
        let configuration = dict(&[
            ("device", c.device.js().clone()),
            ("format", JsValue::from_str(c.format.as_str())),
            ("usage", JsValue::from_f64(f64::from(c.usage))),
            ("viewFormats", view_formats.into()),
            ("colorSpace", JsValue::from_str(c.color_space.as_str())),
            ("toneMapping", tone_mapping),
            ("alphaMode", JsValue::from_str(c.alpha_mode.as_str())),
        ])?;
        call(&self.context, "GPUCanvasContext", "configure", &[configuration])?;
        // A browser that does not implement toneMapping leaves the member out of getConfiguration() (WebGPU §21.4);
        // such a browser shows only the default, so any other mode is refused.
        if c.tone_mapping.mode != CanvasToneMappingMode::Standard {
            let got = call(&self.context, "GPUCanvasContext", "getConfiguration", &[])?;
            let mode = get(&get(&got, "toneMapping"), "mode").as_string();
            if mode.as_deref() != Some(c.tone_mapping.mode.as_str()) {
                let _ = call(&self.context, "GPUCanvasContext", "unconfigure", &[]);
                return Err(Exception::unsupported(format!(
                    "the browser does not implement GPUCanvasConfiguration.toneMapping mode \"{}\"",
                    c.tone_mapping.mode.as_str()
                )));
            }
        }
        Ok(())
    }

    fn unconfigure(&mut self) {
        // unconfigure() has no steps that throw.
        let _ = call(&self.context, "GPUCanvasContext", "unconfigure", &[]);
    }

    fn write_frame(
        &mut self,
        queue: &<Web as crate::gpu::Api>::Queue,
        data: &[u8],
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) -> Result<(), Exception> {
        let texture = call(&self.context, "GPUCanvasContext", "getCurrentTexture", &[])?;
        let destination = dict(&[("texture", texture)])?;
        let layout = dict(&[
            ("offset", JsValue::from_f64(0.0)),
            ("bytesPerRow", JsValue::from_f64(f64::from(bytes_per_row))),
            ("rowsPerImage", JsValue::from_f64(f64::from(height))),
        ])?;
        let size = dict(&[
            ("width", JsValue::from_f64(f64::from(width))),
            ("height", JsValue::from_f64(f64::from(height))),
            ("depthOrArrayLayers", JsValue::from_f64(1.0)),
        ])?;
        let bytes: JsValue = Uint8Array::from(data).into();
        call(queue.js(), "GPUQueue", "writeTexture", &[destination, bytes, layout, size])?;
        Ok(())
    }

    fn present(&mut self) -> Result<(), Exception> {
        Ok(())
    }
}
