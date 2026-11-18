/// <reference lib="webworker" />
// All model inference runs here so the UI thread never freezes.
// Images arrive as ImageBitmaps, are processed in memory and are never sent anywhere.
import * as ort from 'onnxruntime-web/webgpu';
import { decode, INPUT, letterboxParams, toTensorData } from '../lib/yolo';
import { decodeFaces, FACE_INPUT, toFaceTensor } from '../lib/yunet';
import type { Box } from '../lib/matching';
import { PERSON, PERSON_MIN_CONF } from '../lib/classes';

export type WorkerIn =
  | { type: 'init'; id: number; base: string; backend: 'auto' | 'wasm' | 'webgpu' }
  | { type: 'load'; id: number; key: string; url: string; label: string; kind: 'det' | 'face' }
  | { type: 'detect'; id: number; key: string; bitmap: ImageBitmap; conf: number; iou: number; faces: boolean };

const CACHE = 'sitesafe-models-v2'; // entries are keyed by URL including ?v=<content hash>
const sessions = new Map<string, ort.InferenceSession>();
let backend: 'webgpu' | 'wasm' = 'wasm';
let threads = 1;

const post = (msg: unknown) => (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg);

/** Fetch with progress, backed by the Cache API so a second visit loads instantly and offline. */
async function fetchCached(url: string, label: string): Promise<ArrayBuffer> {
  let cache: Cache | null = null;
  try { cache = await caches.open(CACHE); } catch { /* private mode: no cache, still works */ }
  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    post({ type: 'progress', label, loaded: 1, total: 1, cached: true });
    return hit.arrayBuffer();
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not download ${label} (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ type: 'progress', label, loaded, total, cached: false });
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  if (cache) {
    try {
      // drop older versions of the same model so the cache does not grow with every retrain
      const base = url.split('?')[0];
      for (const req of await cache.keys()) if (req.url.split('?')[0] === base && req.url !== url) await cache.delete(req);
      await cache.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(loaded) } }));
    } catch { /* quota: ignore */ }
  }
  return buf.buffer;
}

async function createSession(buf: ArrayBuffer, kind: 'det' | 'face') {
  // The tiny face detector always runs on WASM; the main detector tries WebGPU first.
  if (kind === 'det' && backend === 'webgpu') {
    try {
      return await ort.InferenceSession.create(buf, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' });
    } catch (e) {
      console.warn('WebGPU session failed, falling back to WASM', e);
      backend = 'wasm';
    }
  }
  return ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
}

let detCanvas: OffscreenCanvas | null = null;
let faceCanvas: OffscreenCanvas | null = null;

function letterboxPixels(canvas: OffscreenCanvas, bitmap: ImageBitmap, size: number) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  const lb = letterboxParams(bitmap.width, bitmap.height, size);
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, lb.dx, lb.dy, lb.w, lb.h);
  return { lb, rgba: ctx.getImageData(0, 0, size, size).data };
}

async function detect(msg: Extract<WorkerIn, { type: 'detect' }>) {
  const session = sessions.get(msg.key);
  if (!session) throw new Error(`Model "${msg.key}" is not loaded`);
  const { bitmap } = msg;
  const t0 = performance.now();
  detCanvas ??= new OffscreenCanvas(INPUT, INPUT);
  const { lb, rgba } = letterboxPixels(detCanvas, bitmap, INPUT);
  const input = new ort.Tensor('float32', toTensorData(rgba, INPUT), [1, 3, INPUT, INPUT]);
  const t1 = performance.now();
  const out = await session.run({ [session.inputNames[0]]: input });
  const tensor = out[session.outputNames[0]];
  const data = (await tensor.getData()) as Float32Array;
  const t2 = performance.now();
  // People are kept from a lower confidence than PPE items: a crouching or bent-over worker often scores
  // 0.20-0.25 as a person. Such people show as "check manually" (person_unsure 0.40), never green or red.
  const personConf = Math.min(msg.conf, PERSON_MIN_CONF);
  const dets = decode(data, tensor.dims[2], lb, bitmap.width, bitmap.height, personConf, msg.iou, tensor.dims[1] - 4)
    .filter((d) => d.cls === PERSON || d.conf >= msg.conf);

  let faces: Box[] = [];
  const faceSession = sessions.get('face');
  if (msg.faces && faceSession) {
    faceCanvas ??= new OffscreenCanvas(FACE_INPUT, FACE_INPUT);
    const f = letterboxPixels(faceCanvas, bitmap, FACE_INPUT);
    const fin = new ort.Tensor('float32', toFaceTensor(f.rgba, FACE_INPUT), [1, 3, FACE_INPUT, FACE_INPUT]);
    const fout = await faceSession.run({ [faceSession.inputNames[0]]: fin });
    const named: Record<string, Float32Array> = {};
    for (const name of faceSession.outputNames) named[name] = fout[name].data as Float32Array;
    faces = decodeFaces(named).map(({ box }) => [
      (box[0] - f.lb.dx) / f.lb.ratio, (box[1] - f.lb.dy) / f.lb.ratio,
      (box[2] - f.lb.dx) / f.lb.ratio, (box[3] - f.lb.dy) / f.lb.ratio,
    ] as Box);
  }
  const t3 = performance.now();
  const size = { w: bitmap.width, h: bitmap.height };
  bitmap.close();
  post({
    type: 'result', id: msg.id, dets, faces, ...size,
    ms: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 },
  });
}

self.onmessage = async (e: MessageEvent<WorkerIn>) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      // remove model caches from older app versions
      try {
        for (const k of await caches.keys()) if (k.startsWith('sitesafe-models') && k !== CACHE) await caches.delete(k);
      } catch { /* no Cache API: nothing to clean */ }
      ort.env.wasm.wasmPaths = msg.base + 'ort/';
      threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      ort.env.wasm.numThreads = threads;
      const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
      let hasGpu = false;
      if (msg.backend !== 'wasm' && gpu) {
        try { hasGpu = !!(await gpu.requestAdapter()); } catch { hasGpu = false; }
      }
      backend = hasGpu ? 'webgpu' : 'wasm';
      post({ type: 'done', id: msg.id, backend, threads });
    } else if (msg.type === 'load') {
      if (!sessions.has(msg.key)) {
        const buf = await fetchCached(msg.url, msg.label);
        const session = await createSession(buf, msg.kind);
        if (msg.kind === 'det') {
          // Warm-up: WebGPU compiles its shaders on the first run (seconds). Pay that cost here,
          // behind the loading bar, so the user's first photo is answered at full speed.
          post({ type: 'progress', label: 'Warming up the detector', loaded: 1, total: 1, cached: true });
          const warm = await session.run({
            [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]),
          });
          await warm[session.outputNames[0]].getData();
        }
        sessions.set(msg.key, session);
      }
      post({ type: 'done', id: msg.id, backend, threads });
    } else if (msg.type === 'detect') {
      await detect(msg);
    }
  } catch (err) {
    post({ type: 'error', id: msg.id, message: err instanceof Error ? err.message : String(err) });
  }
};
