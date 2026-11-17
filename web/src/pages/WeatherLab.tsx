// Weather Lab: the generative-AI experiment, made visible. Each real photo was re-rendered by a
// diffusion model as night / rain / fog / dust. Both detectors (trained without and with synthetic
// data) run live on every version and are scored against the real labels.
import { useEffect, useMemo, useRef, useState } from 'react';
import { CloudFog, CloudRain, Loader2, Moon, Sun, Wind } from 'lucide-react';
import { EngineGate } from '../components/EngineStatus';
import { Button, cn, Eyebrow, PageHeader, Panel, Segmented } from '../components/ui';
import { analyse } from '../lib/analyze';
import { blurRegions, drawPeople, overlayFromAnalysis } from '../lib/draw';
import { asset, engine, type ModelKey } from '../lib/engine';
import { iou, type Det } from '../lib/matching';
import { loadCatalogue, urlToCanvas, type WeatherItem } from '../lib/media';

const CONDITIONS = [
  { id: 'original', label: 'Original', icon: Sun },
  { id: 'night', label: 'Night', icon: Moon },
  { id: 'rain', label: 'Rain', icon: CloudRain },
  { id: 'fog', label: 'Fog', icon: CloudFog },
  { id: 'dust', label: 'Dust', icon: Wind },
] as const;

/** Share of labelled boxes the model found (same class, IoU >= 0.5). */
function recall(gt: Det[], dets: Det[]): { found: number; total: number } {
  const used = new Set<number>();
  let found = 0;
  for (const g of gt) {
    let best = -1;
    let bestIou = 0.5;
    dets.forEach((d, i) => {
      if (d.cls !== g.cls || used.has(i)) return;
      const v = iou(d.box, g.box);
      if (v >= bestIou) { best = i; bestIou = v; }
    });
    if (best >= 0) { used.add(best); found += 1; }
  }
  return { found, total: gt.length };
}

type Result = { canvas: HTMLCanvasElement; found: number; total: number; ms: number };

function Tile({ label, icon: Icon, result, other }: { label: string; icon: typeof Sun; result?: Result; other?: Result }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!result || !ref.current) return;
    ref.current.width = result.canvas.width; ref.current.height = result.canvas.height;
    ref.current.getContext('2d')!.drawImage(result.canvas, 0, 0);
  }, [result]);
  const pct = result ? Math.round((100 * result.found) / Math.max(1, result.total)) : null;
  const otherPct = other ? Math.round((100 * other.found) / Math.max(1, other.total)) : null;
  const delta = pct !== null && otherPct !== null ? pct - otherPct : null;
  return (
    <Panel className="overflow-hidden">
      <div className="relative aspect-[4/3] bg-black">
        {result ? <canvas ref={ref} className="h-full w-full object-contain" /> : <div className="grid h-full place-items-center text-muted"><Loader2 className="animate-spin" /></div>}
        <span className="absolute left-2 top-2 flex items-center gap-1 rounded bg-[#0b0d10]/80 px-2 py-0.5 font-mono text-[11px] uppercase text-white"><Icon size={12} /> {label}</span>
      </div>
      <div className="p-3">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-muted">labelled objects found</span>
          <span className="num text-sm font-semibold">{result ? `${result.found}/${result.total}` : '–'}</span>
        </div>
        <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-raised">
          <div className={cn('h-full rounded-full transition-[width] duration-500', (pct ?? 0) >= 75 ? 'bg-ok' : (pct ?? 0) >= 50 ? 'bg-hivis' : 'bg-danger')} style={{ width: `${pct ?? 0}%` }} />
        </div>
        <div className="mt-1 flex justify-between font-mono text-[11px] text-muted">
          <span>{pct ?? '–'}%{delta !== null && delta !== 0 && <span className={delta > 0 ? 'text-ok' : 'text-danger'}> ({delta > 0 ? '+' : ''}{delta} vs other model)</span>}</span>
          <span>{result ? `${result.ms.toFixed(0)} ms` : ''}</span>
        </div>
      </div>
    </Panel>
  );
}

function Lab() {
  const [items, setItems] = useState<WeatherItem[]>([]);
  const [hasSynth, setHasSynth] = useState(false);
  const [sel, setSel] = useState<WeatherItem | null>(null);
  const [model, setModel] = useState<ModelKey>('base');
  const [results, setResults] = useState<Record<string, Result>>({});
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    loadCatalogue().then((c) => { setItems(c.weather); setHasSynth(c.models.synth); setSel(c.weather[0] ?? null); }).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!sel) return;
    let stop = false;
    (async () => {
      setBusy(true);
      const models: ModelKey[] = hasSynth ? [model, model === 'base' ? 'synth' : 'base'] : ['base'];
      for (const m of models) {
        await engine.loadModel(m);
        for (const c of CONDITIONS) {
          const key = `${sel.id}:${m}:${c.id}`;
          if (stop) return;
          if (results[key] || !sel.variants[c.id]) continue;
          const canvas = await urlToCanvas(asset(sel.variants[c.id]));
          const res = await engine.detect(await createImageBitmap(canvas), { model: m, conf: 0.25 });
          const sx = canvas.width / sel.w;
          const gt = sel.gt.map((d) => ({ ...d, box: d.box.map((v) => v * sx) as Det['box'] }));
          const a = analyse(res, 'general', [], { strictBlur: true, forkliftDistance: 0.5 });
          const ctx = canvas.getContext('2d')!;
          blurRegions(ctx, a.blur);
          drawPeople(ctx, overlayFromAnalysis(a), Math.max(1, canvas.width / 700));
          const r = recall(gt, res.dets);
          if (stop) return;
          setResults((prev) => ({ ...prev, [key]: { canvas, ...r, ms: res.ms.total } }));
        }
      }
      setBusy(false);
    })().catch(() => setBusy(false));
    return () => { stop = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel, model, hasSynth]);

  const other: ModelKey = model === 'base' ? 'synth' : 'base';
  const meter = (m: ModelKey) => {
    if (!sel) return null;
    const hard = CONDITIONS.filter((c) => c.id !== 'original').map((c) => results[`${sel.id}:${m}:${c.id}`]).filter(Boolean);
    const orig = results[`${sel.id}:${m}:original`];
    if (hard.length < 4 || !orig) return null;
    const f = hard.reduce((s, r) => s + r.found, 0);
    const t = hard.reduce((s, r) => s + r.total, 0);
    return { hard: Math.round((100 * f) / Math.max(1, t)), orig: Math.round((100 * orig.found) / Math.max(1, orig.total)) };
  };
  const mA = useMemo(() => meter(model), [results, sel, model]); // eslint-disable-line react-hooks/exhaustive-deps
  const mB = useMemo(() => (hasSynth ? meter(other) : null), [results, sel, other, hasSynth]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <PageHeader eyebrow="Weather Lab · generative AI" title="How does the detector cope with night, rain, fog and dust?">
        A diffusion model (SD-Turbo, image-to-image at low strength) re-rendered each real photo in four hard conditions. Positions do not move, so the original labels still apply and the model can be scored on each version.
      </PageHeader>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex gap-2 overflow-x-auto">
          {items.map((it) => (
            <button key={it.id} type="button" onClick={() => setSel(it)} className={cn('h-16 w-24 shrink-0 overflow-hidden rounded border', sel?.id === it.id ? 'border-hivis' : 'border-line')}>
              <img src={asset(it.variants.original.replace('.jpg', '_thumb.jpg'))} alt={it.title} className="h-full w-full object-cover" />
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          {busy && <Loader2 size={15} className="animate-spin text-muted" />}
          {hasSynth ? (
            <Segmented<ModelKey> label="Detector" value={model} onChange={setModel}
              options={[{ value: 'base', label: 'Trained on real photos only' }, { value: 'synth', label: 'Real + synthetic' }]} />
          ) : <span className="rounded border border-line px-3 py-1.5 text-xs text-muted">Detector: real photos only (synthetic-trained model not shipped yet)</span>}
        </div>
      </div>

      {!items.length && <Panel className="p-6 text-sm text-muted">Weather samples are not built yet. Run <code className="font-mono">python scripts/build_web_assets.py</code>.</Panel>}

      {sel && (
        <>
          <Panel className="mb-4 grid gap-4 p-4 sm:grid-cols-[1fr_auto] sm:items-center">
            <div>
              <Eyebrow>Robustness meter</Eyebrow>
              <p className="mt-1 text-sm text-muted">Share of labelled objects still found across the four hard versions of this photo.</p>
              <div className="mt-3 space-y-2">
                {[{ m: model, v: mA }, ...(hasSynth ? [{ m: other, v: mB }] : [])].map(({ m, v }) => (
                  <div key={m} className="flex items-center gap-3">
                    <span className="w-36 shrink-0 text-xs">{m === 'base' ? 'Real only' : 'Real + synthetic'}</span>
                    <div className="h-3 flex-1 overflow-hidden rounded-full bg-raised">
                      <div className={cn('h-full rounded-full transition-[width] duration-700', m === model ? 'bg-hivis' : 'bg-muted')} style={{ width: `${v?.hard ?? 0}%` }} />
                    </div>
                    <span className="num w-28 text-right text-xs">{v ? `${v.hard}% (clear: ${v.orig}%)` : 'measuring…'}</span>
                  </div>
                ))}
              </div>
            </div>
            {mA && mB && (
              <div className="rounded border border-line px-4 py-3 text-center">
                <div className={cn('num text-3xl font-bold', (model === 'synth' ? mA.hard - mB.hard : mB.hard - mA.hard) >= 0 ? 'text-ok' : 'text-danger')}>
                  {(model === 'synth' ? mA.hard - mB.hard : mB.hard - mA.hard) >= 0 ? '+' : ''}{model === 'synth' ? mA.hard - mB.hard : mB.hard - mA.hard}
                </div>
                <div className="eyebrow">points from synthetic data<br />on this photo</div>
              </div>
            )}
          </Panel>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {CONDITIONS.map((c) => (
              <Tile key={c.id} label={c.label} icon={c.icon} result={results[`${sel.id}:${model}:${c.id}`]} other={hasSynth ? results[`${sel.id}:${other}:${c.id}`] : undefined} />
            ))}
          </div>
          <p className="mt-4 text-xs text-muted">
            One photo is an illustration, not evidence. The measured effect on the whole held-out test set (with confidence intervals) is on the Model card page and in the report. Synthetic images are used for training and this demo only; the test set contains real photos.
          </p>
          <div className="mt-3"><Button size="sm" onClick={() => setResults({})}>Run again</Button></div>
        </>
      )}
    </div>
  );
}

export default function WeatherLab() {
  return <EngineGate><Lab /></EngineGate>;
}
