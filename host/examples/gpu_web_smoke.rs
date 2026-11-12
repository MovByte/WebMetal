//! The conformance smoke test (`webmetal_host::gpu::smoke`) in a browser: built for wasm32 with the `web` feature and
//! loaded by `host/tools/gpu_web_smoke.py`, which serves the page, runs it in a headless browser and collects the
//! report the page posts back.

#[cfg(target_arch = "wasm32")]
mod page {
    use wasm_bindgen::prelude::*;
    use webmetal_host::gpu::smoke;
    use webmetal_host::gpu::web::{Web, WebGpu};

    /// Run the smoke test and resolve with its report text.
    #[wasm_bindgen]
    pub async fn run_smoke() -> Result<String, JsValue> {
        let gpu = WebGpu::from_global().map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(smoke::run::<Web>(&gpu).await.text())
    }
}
