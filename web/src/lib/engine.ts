// Main-thread handle on the inference worker.
import { create } from 'zustand';
import type { Box, Det } from './matching';

export interface DetectResult {
  dets: Det[];
  faces: Box[];
  w: number;
  h: number;
  ms: { pre: number; infer: number; post: number; total: number };
}

export type ModelKey = 'base' | 'synth';
export const MODEL_FILES: Record<ModelKey, { file: string; label: string }> = {
  base: { file: 'models/sitesafe_base.onnx', label: 'PPE detector' },
  synth: { file: 'models/sitesafe_synth.onnx', label: 'PPE detector (trained with synthetic data)' },
};

interface EngineState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  backend: 'webgpu' | 'wasm' | null;
  threads: number;
  progress: { label: string; pct: number; cached: boolean } | null;
  error: string | null;
  loaded: Record<string, boolean>;
}
export const useEngine = create<EngineState>(() => ({
  status: 'idle', backend: null, threads: 1, progress: null, error: null, loaded: {},
}));

export const BASE = import.meta.env.BASE_URL;
export const asset = (path: string) => new URL(BASE + path, window.location.href).toString();

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void };

class Engine {
  private worker: Worker | null = null;
  private pending = new Map<number, Pending>();
  private seq = 0;
  private initPromise: Promise<void> | null = null;
  private loading = new Map<string, Promise<void>>();

  private call<T>(msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const id = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage({ ...msg, id }, transfer);
    });
  }

  private boot() {
    this.worker = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') {
        useEngine.setState({
          progress: { label: m.label, pct: m.total ? Math.round((100 * m.loaded) / m.total) : 0, cached: m.cached },
        });
        return;
      }
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      if (m.type === 'error') p.reject(new Error(m.message));
      else p.resolve(m);
    };
    this.worker.onerror = (e) => {
      useEngine.setState({ status: 'error', error: e.message || 'The inference worker crashed.' });
      for (const p of this.pending.values()) p.reject(new Error(e.message));
      this.pending.clear();
    };
  }

  /** Load the runtime, the detector and the face detector. Safe to call many times. */
  init(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    useEngine.setState({ status: 'loading', error: null });
    this.initPromise = (async () => {
      try {
        this.boot();
        const pref = (localStorage.getItem('sitesafe-backend') as 'auto' | 'wasm' | 'webgpu') || 'auto';
        await this.call({ type: 'init', base: asset(''), backend: pref });
        await this.loadModel('base');
        const info = await this.call<{ backend: 'webgpu' | 'wasm'; threads: number }>({
          type: 'load', key: 'face', url: asset('models/yunet.onnx'), label: 'Face detector', kind: 'face',
        });
        useEngine.setState({ status: 'ready', backend: info.backend, threads: info.threads, progress: null });
      } catch (err) {
        this.initPromise = null;
        useEngine.setState({ status: 'error', error: err instanceof Error ? err.message : String(err) });
        throw err;
      }
    })();
    return this.initPromise;
  }

  loadModel(key: ModelKey): Promise<void> {
    let p = this.loading.get(key);
    if (!p) {
      p = this.call({ type: 'load', key, url: asset(MODEL_FILES[key].file), label: MODEL_FILES[key].label, kind: 'det' })
        .then(() => { useEngine.setState((s) => ({ loaded: { ...s.loaded, [key]: true }, progress: null })); });
      p.catch(() => this.loading.delete(key));
      this.loading.set(key, p);
    }
    return p;
  }

  async detect(
    source: ImageBitmapSource, opts: { conf?: number; iou?: number; faces?: boolean; model?: ModelKey } = {},
  ): Promise<DetectResult> {
    await this.init();
    const bitmap = source instanceof ImageBitmap ? source : await createImageBitmap(source);
    return this.call<DetectResult>({
      type: 'detect', key: opts.model ?? 'base', bitmap, conf: opts.conf ?? 0.25, iou: opts.iou ?? 0.5,
      faces: opts.faces ?? true,
    }, [bitmap]);
  }
}

export const engine = new Engine();
