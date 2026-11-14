// One case in a fresh instance of its module: `{ glue, module, index }` runs case `index` and answers its outcome;
// `{ glue, module, list: true }` answers the module's case names.
self.onmessage = async (e) => {
  const { glue, module, index, list } = e.data;
  globalThis.__wmCase = { log: [], panic: null, names: [] };
  const c = globalThis.__wmCase;
  try {
    const { initSync } = await import(glue);
    const wasm = initSync({ module });
    if (list) {
      const n = wasm.wm_case_count();
      for (let i = 0; i < n; i++) wasm.wm_case_name(i);
      postMessage({ status: 'listed', names: c.names });
      return;
    }
    const run = WebAssembly.promising(wasm.wm_run_case);
    await run(index);
    postMessage({ status: 'pass', log: c.log });
  } catch (err) {
    postMessage({ status: 'fail', panic: c.panic, error: String(err && err.stack ? err.stack : err), log: c.log });
  }
};
