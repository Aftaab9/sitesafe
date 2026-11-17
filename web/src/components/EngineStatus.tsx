import { useEffect } from 'react';
import { Cpu, Loader2, TriangleAlert, Zap } from 'lucide-react';
import { engine, useEngine } from '../lib/engine';
import { Button, Panel } from './ui';

/** Small header pill: which backend the model is running on. */
export function EngineBadge() {
  const { status, backend, threads } = useEngine();
  if (status === 'idle') return null;
  if (status === 'loading') {
    return (
      <span className="hidden items-center gap-1.5 whitespace-nowrap rounded-full border border-line px-2.5 py-1 font-mono text-[11px] text-muted xl:flex">
        <Loader2 size={12} className="animate-spin" /> loading model
      </span>
    );
  }
  if (status === 'error') {
    return (
      <span className="hidden items-center gap-1.5 whitespace-nowrap rounded-full border border-danger/50 px-2.5 py-1 font-mono text-[11px] text-danger xl:flex">
        <TriangleAlert size={12} /> model error
      </span>
    );
  }
  return (
    <span
      className="hidden items-center gap-1.5 whitespace-nowrap rounded-full border border-line px-2.5 py-1 font-mono text-[11px] text-muted xl:flex"
      title={backend === 'webgpu' ? 'Running on your GPU via WebGPU' : `Running on your CPU via WebAssembly (${threads} thread${threads > 1 ? 's' : ''})`}
    >
      {backend === 'webgpu' ? <Zap size={12} className="text-hivis" /> : <Cpu size={12} className="text-hivis" />}
      on-device · {backend === 'webgpu' ? 'WebGPU' : `WASM×${threads}`}
    </span>
  );
}

const COPY = [
  'Unpacking the hard hats…',
  'Checking the hi-vis…',
  'Lacing up the boots…',
  'Briefing the detector…',
];

/** Blocks a page until the model is loaded. Shows download progress with friendly copy. */
export function EngineGate({ children }: { children: React.ReactNode }) {
  const { status, progress, error } = useEngine();
  useEffect(() => { engine.init().catch(() => undefined); }, []);
  if (status === 'ready') return <>{children}</>;
  if (status === 'error') {
    return (
      <Panel className="mx-auto mt-10 max-w-lg p-6 text-center">
        <TriangleAlert className="mx-auto text-danger" />
        <h2 className="mt-3 text-lg font-semibold">The model could not start</h2>
        <p className="mt-2 text-sm text-muted">{error}</p>
        <p className="mt-2 text-sm text-muted">
          Check your connection for the first load (afterwards the app works offline), or try the CPU engine.
        </p>
        <div className="mt-4 flex justify-center gap-2">
          <Button variant="primary" onClick={() => location.reload()}>Try again</Button>
          <Button onClick={() => { localStorage.setItem('sitesafe-backend', 'wasm'); location.reload(); }}>Use CPU engine</Button>
        </div>
      </Panel>
    );
  }
  const pct = progress?.pct ?? 0;
  return (
    <Panel className="mx-auto mt-10 max-w-lg p-6" role="status" aria-live="polite">
      <div className="eyebrow">Getting ready</div>
      <h2 className="mt-1 text-lg font-semibold">{COPY[Math.floor(pct / 26) % COPY.length]}</h2>
      <p className="mt-1 text-sm text-muted">
        {progress?.cached
          ? 'Loading the model from this device.'
          : 'Downloading the model to this device (about 11 MB, once). After that it works offline.'}
      </p>
      <div className="mt-4 h-3 overflow-hidden rounded-full border border-line bg-raised">
        <div className="h-full hazard-stripes animate-stripes transition-[width] duration-200" style={{ width: `${Math.max(4, pct)}%` }} />
      </div>
      <div className="mt-2 flex justify-between font-mono text-xs text-muted">
        <span>{progress?.label ?? 'Starting the on-device engine'}</span>
        <span>{pct}%</span>
      </div>
    </Panel>
  );
}
