//! The presentation sink on a page's canvas: built for wasm32 with the `web` feature and loaded by
//! `host/embed-ref/tools/present_web.py`, which serves the page, runs it in a headless browser and collects the report
//! the page posts back.
//!
//! For each case the page attaches a display shown on a canvas in the document, fills an IOSurface from the CPU and
//! presents it. The display's target reads back what the browser shows when the sink presents the frame: the canvas
//! drawn into a 2D canvas of the same color space and read with `getImageData()`, which is "get a copy of the image
//! contents of a context" (WebGPU §21.3). The red, green and blue codes must equal the surface's, and alpha must be
//! 255 (the display is opaque).
//!
//! The same read after two rendering updates is reported as an observation, not a check: Chrome 157 returns
//! transparent black for a WebGPU canvas whose current texture has expired, however it was written (render pass,
//! copy or `writeTexture()`, measured), where WebGPU §21.2 keeps the drawing buffer until it is replaced.

#[cfg(target_arch = "wasm32")]
mod page {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use js_sys::{Array, Function, Object, Promise, Reflect, Uint8ClampedArray};
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use webmetal_embed_ref::{Clock, RefEmbedder, RefHost};
    use webmetal_host::embed::{PresentSource, PresentedFrame, Rect, SurfaceDesc};
    use webmetal_host::exec::Config;
    use webmetal_host::gpu::idl::CanvasConfiguration;
    use webmetal_host::gpu::idl::RequestAdapterOptions;
    use webmetal_host::gpu::scope::{Granularity, Policy};
    use webmetal_host::gpu::web::{Web, WebGpu};
    use webmetal_host::gpu::{Adapter as _, Device as _, Exception, Gpu as _, negotiate};
    use webmetal_host::present::format::code;
    use webmetal_host::present::web::CanvasTarget;
    use webmetal_host::present::{DisplayColorSpace, DisplayMode, SwapRequest, Target};
    use webmetal_wire::SurfaceId;

    fn err(e: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&e.to_string())
    }

    fn get(o: &JsValue, k: &str) -> JsValue {
        Reflect::get(o, &JsValue::from_str(k)).unwrap_or(JsValue::UNDEFINED)
    }

    fn call(o: &JsValue, m: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
        let f: Function = get(o, m).dyn_into()?;
        let a = Array::new();
        for x in args {
            a.push(x);
        }
        Reflect::apply(&f, o, &a)
    }

    fn dict(entries: &[(&str, JsValue)]) -> Result<JsValue, JsValue> {
        let o: JsValue = Object::new().into();
        for (k, v) in entries {
            Reflect::set(&o, &JsValue::from_str(k), v)?;
        }
        Ok(o)
    }

    /// Resolves at the next animation frame callback.
    async fn animation_frame() -> Result<(), JsValue> {
        let p = Promise::new(&mut |resolve, _| {
            let _ = call(&js_sys::global(), "requestAnimationFrame", &[resolve.into()]);
        });
        JsFuture::from(p).await.map(|_| ())
    }

    fn pattern(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    /// What the browser shows of `canvas`, as RGBA bytes read through a 2D canvas in `color_space`.
    fn shown(document: &JsValue, canvas: &JsValue, color_space: &str, w: u32, h: u32) -> Result<Vec<u8>, JsValue> {
        let c2 = call(document, "createElement", &[JsValue::from_str("canvas")])?;
        Reflect::set(&c2, &"width".into(), &JsValue::from(w))?;
        Reflect::set(&c2, &"height".into(), &JsValue::from(h))?;
        let settings = dict(&[("colorSpace", JsValue::from_str(color_space)), ("willReadFrequently", JsValue::TRUE)])?;
        let ctx = call(&c2, "getContext", &[JsValue::from_str("2d"), settings])?;
        call(&ctx, "drawImage", &[canvas.clone(), JsValue::from(0), JsValue::from(0)])?;
        let opts = dict(&[("colorSpace", JsValue::from_str(color_space))])?;
        let image = call(&ctx, "getImageData", &[0.into(), 0.into(), w.into(), h.into(), opts])?;
        let data: Uint8ClampedArray = get(&image, "data").dyn_into()?;
        Ok(data.to_vec())
    }

    /// "exact", or how the bytes differ.
    fn verdict(got: &[u8], want: &[u8]) -> String {
        if got.len() != want.len() {
            return format!("{} bytes, {} expected", got.len(), want.len());
        }
        let diff: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
        match diff.first() {
            None => "exact".into(),
            Some(&i) => format!("{} bytes differ, first at {i}: {} for {}", diff.len(), got[i], want[i]),
        }
    }

    /// What the browser showed at each presentation, or why it could not be read.
    type Reads = Rc<RefCell<Vec<Result<Vec<u8>, String>>>>;

    /// A canvas target that reads back what the browser shows each time a frame is presented, in the task that wrote
    /// it.
    struct Capturing {
        canvas: CanvasTarget,
        document: JsValue,
        color_space: &'static str,
        shown: Reads,
    }

    impl Target<Web> for Capturing {
        fn size(&self) -> (u32, u32) {
            self.canvas.size()
        }
        fn set_size(&mut self, width: u32, height: u32) {
            self.canvas.set_size(width, height)
        }
        fn configure(&mut self, configuration: &CanvasConfiguration<'_, Web>) -> Result<(), Exception> {
            self.canvas.configure(configuration)
        }
        fn unconfigure(&mut self) {
            self.canvas.unconfigure()
        }
        fn write_frame(
            &mut self,
            queue: &<Web as webmetal_host::gpu::Api>::Queue,
            data: &[u8],
            bytes_per_row: u32,
            width: u32,
            height: u32,
        ) -> Result<(), Exception> {
            self.canvas.write_frame(queue, data, bytes_per_row, width, height)
        }
        fn present(&mut self) -> Result<(), Exception> {
            self.canvas.present()?;
            let (w, h) = self.canvas.size();
            let canvas = get(self.canvas.context(), "canvas");
            let read = shown(&self.document, &canvas, self.color_space, w, h).map_err(|e| format!("{e:?}"));
            self.shown.borrow_mut().push(read);
            Ok(())
        }
    }

    /// Run the cases and resolve with the report text.
    #[wasm_bindgen]
    pub async fn run_present() -> Result<String, JsValue> {
        let mut report = String::new();
        let gpu = WebGpu::from_global().map_err(err)?;
        let adapter = gpu.request_adapter(&RequestAdapterOptions::new()).await.map_err(err)?.ok_or("no adapter")?;
        let info = adapter.info();
        let field = |v: &Option<String>| v.clone().unwrap_or_default();
        report.push_str(&format!(
            "info adapter = {} {} {}\n",
            field(&info.vendor),
            field(&info.architecture),
            field(&info.description)
        ));
        let mut d = match negotiate::plan(&adapter.features(), &adapter.limits()) {
            Ok(p) => p.descriptor,
            Err(e) => return Err(err(format!("negotiation: {e}"))),
        };
        d.label = "present page device".into();
        let device = adapter.request_device(&d).await.map_err(err)?;
        let embedder = Rc::new(RefEmbedder::new(Clock::Manual(Cell::new(0))));
        let config = Config { scopes: Policy { granularity: Granularity::PerCall, strict: true } };
        let host = RefHost::<Web>::new(device.clone(), embedder.clone(), config).await.map_err(err)?;
        let document = get(&js_sys::global(), "document");
        let body = get(&document, "body");

        let (w, h, bpr) = (64u32, 32u32, 320u64);
        for (display, (space, name)) in
            [(DisplayColorSpace::Srgb, "srgb"), (DisplayColorSpace::DisplayP3, "display-p3")].into_iter().enumerate()
        {
            let display = display as u32;
            let canvas = call(&document, "createElement", &[JsValue::from_str("canvas")])?;
            call(&body, "appendChild", std::slice::from_ref(&canvas))?;
            let context = call(&canvas, "getContext", &[JsValue::from_str("webgpu")])?;
            let captured = Rc::new(RefCell::new(Vec::new()));
            let target = Capturing {
                canvas: CanvasTarget::new(context).map_err(err)?,
                document: document.clone(),
                color_space: name,
                shown: captured.clone(),
            };
            let mode = DisplayMode { width: w, height: h, color_space: space, identity_transform: 0 };
            host.presenter.attach(display, mode, Box::new(target)).map_err(|f| err(f.failure.message))?;
            embedder.add_display(display);
            let id = 10 + display;
            embedder
                .create_surface(SurfaceDesc {
                    id: SurfaceId(id),
                    width: w.into(),
                    height: h.into(),
                    bytes_per_row: bpr,
                    bytes_per_element: 4,
                    element_width: 1,
                    element_height: 1,
                    pixel_format: code::BGRA_32,
                    alloc_size: bpr * u64::from(h),
                    planes: vec![],
                    seed: 1,
                })
                .map_err(err)?;
            let pixels = pattern((bpr * u64::from(h)) as usize, 0x1234_5677 + id);
            embedder.surface_cpu_write(SurfaceId(id), 0, &pixels).map_err(err)?;
            let rect = Rect { x: 0, y: 0, width: w.into(), height: h.into() };
            let request = SwapRequest {
                frame: PresentedFrame {
                    display,
                    source: PresentSource::Surface { id: SurfaceId(id), plane: 0 },
                    source_rect: rect,
                    target_rect: rect,
                    transform: 0,
                    time: None,
                },
                minimum_duration: None,
            };
            let p = host.presenter.present(request).await.map_err(|f| err(f.failure.message))?;
            report.push_str(&format!(
                "info present.{name} = {} {}x{} {}\n",
                p.plan.format.as_str(),
                p.plan.width,
                p.plan.height,
                p.plan.color_space
            ));
            // The surface's blue, green, red codes as red, green, blue; alpha 255.
            let mut want = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let s = &pixels[y * bpr as usize + x * 4..][..4];
                    want.extend_from_slice(&[s[2], s[1], s[0], 0xff]);
                }
            }
            let at_present = match captured.borrow().as_slice() {
                [Ok(b)] => verdict(b, &want),
                other => format!("{} reads: {other:?}", other.len()),
            };
            report.push_str(&format!("check present.{name}.at-presentation = {at_present}\n"));
            animation_frame().await?;
            animation_frame().await?;
            let later = shown(&document, &canvas, name, w, h)?;
            report.push_str(&format!("observe present.{name}.after-rendering-updates = {}\n", verdict(&later, &want)));
        }
        device.destroy();
        Ok(report)
    }
}
