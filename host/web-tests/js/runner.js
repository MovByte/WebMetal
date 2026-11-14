// The page side of webmetal-web-tests (src/lib.rs): the suspending waits of the blocking GPU, and the case's output.
// The Worker that runs a case sets globalThis.__wmCase ({ log: [], panic: null, names: [] }) before it instantiates
// the module.

let woken = false;
let settle = null;

export const wm_suspend = new WebAssembly.Suspending((ms) => new Promise((resolve) => {
  if (woken) {
    woken = false;
    resolve(1);
    return;
  }
  const timer = setTimeout(() => { settle = null; resolve(0); }, ms);
  settle = () => { clearTimeout(timer); settle = null; resolve(1); };
}));

export function wm_wake() {
  if (settle) settle();
  else woken = true;
}

export const wm_yield = new WebAssembly.Suspending(() => new Promise((resolve) => setTimeout(resolve, 0)));

export function wm_log(line) {
  globalThis.__wmCase.log.push(line);
}

export function wm_panic(message) {
  globalThis.__wmCase.panic = message;
}

export function wm_case(index, name) {
  globalThis.__wmCase.names[index] = name;
}

// Synchronous requests are allowed in a Worker, with any response type.
function get(url, type) {
  const x = new XMLHttpRequest();
  x.open('GET', url, false);
  x.responseType = type;
  x.send();
  return x.status === 200 ? x.response : undefined;
}

export function wm_read(path) {
  const r = get('/' + path, 'arraybuffer');
  return r === undefined ? undefined : new Uint8Array(r);
}

export function wm_list(path) {
  return get('/__list/' + path, 'text');
}
